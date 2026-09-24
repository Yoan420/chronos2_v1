"""Leakage-safe hourly features on the canonical UTC delivery timeline.

This module assumes covariates have already been selected from point-in-time
vintages by the upstream loader.  It does not resample, interpolate, forward
fill, or backward fill any input.  Price-derived features respect the joint
day-ahead decision: every hour of local delivery day D may only use target
prices belonging to a strictly earlier local delivery day.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd


DEFAULT_PRICE_LAGS: tuple[int, ...] = (24, 48, 168)
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (24, 168)
DEFAULT_ROLLING_STATISTICS: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "max",
)
SUPPORTED_ROLLING_STATISTICS = frozenset(
    {"mean", "std", "min", "max", "median"}
)
FeatureScope = Literal["all", "future"]


class HourlyFeatureContractError(ValueError):
    """Raised when an input can no longer guarantee causal hourly features."""


def validate_utc_hourly_index(
    index: pd.Index,
    *,
    name: str = "index",
    require_contiguous: bool = True,
) -> pd.DatetimeIndex:
    """Validate the canonical sorted, unique, hourly UTC index contract.

    The function validates without sorting or coercing.  This is intentional:
    silently repairing the index could change the meaning of positional lags.
    """

    if not isinstance(index, pd.DatetimeIndex):
        raise HourlyFeatureContractError(
            f"{name} doit être un pandas.DatetimeIndex."
        )
    if index.tz is None or str(index.tz).upper() != "UTC":
        raise HourlyFeatureContractError(
            f"{name} doit utiliser explicitement le fuseau UTC."
        )
    if index.has_duplicates:
        duplicated = index[index.duplicated()].unique()
        raise HourlyFeatureContractError(
            f"{name} contient {len(duplicated)} timestamp(s) dupliqué(s)."
        )
    if not index.is_monotonic_increasing:
        raise HourlyFeatureContractError(
            f"{name} doit être strictement croissant."
        )

    aligned = (
        (index.minute == 0)
        & (index.second == 0)
        & (index.microsecond == 0)
        & (index.nanosecond == 0)
    )
    if not bool(np.all(aligned)):
        examples = [str(value) for value in index[~aligned][:3]]
        raise HourlyFeatureContractError(
            f"{name} contient des timestamps non horaires: {examples}."
        )

    if require_contiguous and len(index) > 1:
        differences = index[1:] - index[:-1]
        invalid = differences != pd.Timedelta(hours=1)
        if bool(np.any(invalid)):
            positions = np.flatnonzero(invalid)[:3]
            examples = [
                f"{index[position]} -> {index[position + 1]}"
                for position in positions
            ]
            raise HourlyFeatureContractError(
                f"{name} n'est pas continu au pas horaire UTC: {examples}."
            )
    return index


def _normalise_positive_hours(
    values: Sequence[int],
    *,
    name: str,
) -> tuple[int, ...]:
    result: list[int] = []
    for raw_value in values:
        if isinstance(raw_value, (bool, np.bool_)):
            raise HourlyFeatureContractError(
                f"{name} doit contenir des entiers strictement positifs."
            )
        try:
            value = int(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise HourlyFeatureContractError(
                f"{name} doit contenir des entiers strictement positifs."
            ) from exc
        if value <= 0 or value != raw_value:
            raise HourlyFeatureContractError(
                f"{name} doit contenir des entiers strictement positifs."
            )
        if value not in result:
            result.append(value)
    return tuple(result)


def _normalise_rolling_statistics(
    statistics: Sequence[str],
) -> tuple[str, ...]:
    result: list[str] = []
    for raw_statistic in statistics:
        statistic = str(raw_statistic).strip().lower()
        if statistic not in SUPPORTED_ROLLING_STATISTICS:
            raise HourlyFeatureContractError(
                "rolling_statistics contient une statistique inconnue: "
                f"{raw_statistic!r}. Valeurs admises: "
                f"{sorted(SUPPORTED_ROLLING_STATISTICS)}."
            )
        if statistic not in result:
            result.append(statistic)
    return tuple(result)


def _numeric_target(target: pd.Series, *, name: str) -> pd.Series:
    if not isinstance(target, pd.Series):
        raise TypeError(f"{name} doit être une pandas.Series.")
    validate_utc_hourly_index(target.index, name=f"{name}.index")
    try:
        numeric = pd.to_numeric(target, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise HourlyFeatureContractError(
            f"{name} doit contenir uniquement des prix numériques."
        ) from exc
    if bool(np.isinf(numeric.to_numpy(dtype=float, copy=False)).any()):
        raise HourlyFeatureContractError(
            f"{name} contient des prix infinis."
        )
    numeric = numeric.copy()
    numeric.name = target.name or "price"
    return numeric


def _validate_covariates(
    covariates: pd.DataFrame,
    *,
    expected_index: pd.DatetimeIndex,
    name: str,
) -> pd.DataFrame:
    if not isinstance(covariates, pd.DataFrame):
        raise TypeError(f"{name} doit être un pandas.DataFrame.")
    validate_utc_hourly_index(covariates.index, name=f"{name}.index")
    if not covariates.index.equals(expected_index):
        raise HourlyFeatureContractError(
            f"{name}.index doit être exactement égal à l'index cible; "
            "aucun reindexage ou remplissage implicite n'est autorisé."
        )
    if not covariates.columns.is_unique:
        duplicates = covariates.columns[
            covariates.columns.duplicated()
        ].tolist()
        raise HourlyFeatureContractError(
            f"{name} contient des colonnes dupliquées: {duplicates}."
        )
    return covariates.copy(deep=True)


def build_calendar_features(
    index: pd.DatetimeIndex,
    *,
    timezone: str = "Europe/Paris",
) -> pd.DataFrame:
    """Create numeric local-calendar metadata on a canonical UTC index."""

    validate_utc_hourly_index(index, name="calendar.index")
    try:
        local = index.tz_convert(timezone)
    except (TypeError, ValueError, KeyError) as exc:
        raise HourlyFeatureContractError(
            f"Fuseau de livraison invalide: {timezone!r}."
        ) from exc

    local_python = local.to_pydatetime()
    hour = local.hour.to_numpy(dtype=np.int16)
    weekday = local.dayofweek.to_numpy(dtype=np.int8)
    day_of_year = local.dayofyear.to_numpy(dtype=np.int16)
    fold = np.asarray([value.fold for value in local_python], dtype=np.int8)
    offset_hours = np.asarray(
        [value.utcoffset().total_seconds() / 3600.0 for value in local_python],
        dtype=float,
    )
    is_dst = np.asarray(
        [value.dst().total_seconds() != 0.0 for value in local_python],
        dtype=np.int8,
    )
    hour_angle = 2.0 * np.pi * hour / 24.0
    weekday_angle = 2.0 * np.pi * weekday / 7.0
    year_angle = 2.0 * np.pi * (day_of_year - 1) / 365.2425

    return pd.DataFrame(
        {
            "calendar_local_hour": hour,
            "calendar_weekday": weekday,
            "calendar_is_weekend": (weekday >= 5).astype(np.int8),
            "calendar_hour_sin": np.sin(hour_angle),
            "calendar_hour_cos": np.cos(hour_angle),
            "calendar_weekday_sin": np.sin(weekday_angle),
            "calendar_weekday_cos": np.cos(weekday_angle),
            "calendar_dayofyear_sin": np.sin(year_angle),
            "calendar_dayofyear_cos": np.cos(year_angle),
            "calendar_dst_fold": fold,
            "calendar_is_dst": is_dst,
            "calendar_utc_offset_hours": offset_hours,
        },
        index=index.copy(),
    )


def _price_history_features(
    target: pd.Series,
    *,
    price_lags: tuple[int, ...],
    rolling_windows: tuple[int, ...],
    rolling_statistics: tuple[str, ...],
    timezone: str,
) -> pd.DataFrame:
    features = pd.DataFrame(index=target.index.copy())
    local_index = target.index.tz_convert(timezone)
    local_days = pd.Series(
        [value.date() for value in local_index.to_pydatetime()],
        index=target.index,
        name="local_delivery_day",
    )
    day_codes = pd.Series(
        pd.factorize(local_days, sort=False)[0],
        index=target.index,
        dtype="int64",
    )

    for lag in price_lags:
        lagged_price = target.shift(lag)
        source_day_code = day_codes.shift(lag)
        # A physical lag of 24 hours points back into D itself for the 25th
        # hour of an autumn DST day.  Masking by local delivery day is the
        # necessary day-ahead causality check, not an optional DST fix.
        strictly_earlier_day = (
            source_day_code.notna() & source_day_code.lt(day_codes)
        )
        features[f"price_lag_{lag}h"] = lagged_price.where(
            strictly_earlier_day
        )

    # At the first row of D, shift(1) ends exactly at D-1's final local hour.
    # The value calculated there is then broadcast over all rows of D.  Taking
    # a positional first value is important: groupby.first would skip an
    # initial NaN and could select a later, same-day (leaking) statistic.
    past_only = target.shift(1)
    day_values = local_days.to_numpy()
    day_start_positions = np.r_[
        0,
        np.flatnonzero(day_values[1:] != day_values[:-1]) + 1,
    ]
    day_end_positions = np.r_[day_start_positions[1:], len(target)]
    for window in rolling_windows:
        rolling = past_only.rolling(window=window, min_periods=window)
        for statistic in rolling_statistics:
            column = f"price_rolling_{statistic}_{window}h"
            if statistic == "std":
                raw_statistic = rolling.std(ddof=0)
            else:
                raw_statistic = getattr(rolling, statistic)()
            frozen_values = np.full(len(target), np.nan, dtype=float)
            for start, end in zip(day_start_positions, day_end_positions):
                frozen_values[start:end] = raw_statistic.iloc[start]
            features[column] = frozen_values
    return features


def _combine_feature_blocks(
    blocks: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    all_columns: list[object] = []
    for block in blocks:
        all_columns.extend(block.columns.tolist())
    duplicated = pd.Index(all_columns)[pd.Index(all_columns).duplicated()]
    if len(duplicated):
        names = list(dict.fromkeys(str(value) for value in duplicated))
        raise HourlyFeatureContractError(
            "Collision entre covariables et features générées: "
            f"{names}."
        )
    return pd.concat(blocks, axis=1, copy=False)


def build_hourly_feature_matrix(
    target: pd.Series,
    covariates: pd.DataFrame,
    *,
    price_lags: Sequence[int] = DEFAULT_PRICE_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    rolling_statistics: Sequence[str] = DEFAULT_ROLLING_STATISTICS,
    timezone: str = "Europe/Paris",
) -> pd.DataFrame:
    """Build covariate, calendar, lag, and day-ahead-safe price features.

    ``target`` is never copied into the result at its contemporaneous value.
    Missing targets and covariates remain missing.  Because the UTC index is
    required to be continuous, a positional ``shift(24)`` is exactly a
    24-hour physical-time lag.  The source local day is checked afterwards so
    that the 25th autumn hour cannot read the first price of that same day.
    Rolling statistics are evaluated at the first instant of D using data up
    to D-1 23:00 local, then held constant over all hours of D.
    """

    numeric_target = _numeric_target(target, name="target")
    clean_covariates = _validate_covariates(
        covariates,
        expected_index=numeric_target.index,
        name="covariates",
    )
    lags = _normalise_positive_hours(price_lags, name="price_lags")
    windows = _normalise_positive_hours(
        rolling_windows,
        name="rolling_windows",
    )
    statistics = _normalise_rolling_statistics(rolling_statistics)

    calendar = build_calendar_features(
        numeric_target.index,
        timezone=timezone,
    )
    prices = _price_history_features(
        numeric_target,
        price_lags=lags,
        rolling_windows=windows,
        rolling_statistics=statistics,
        timezone=timezone,
    )
    result = _combine_feature_blocks(
        [clean_covariates, calendar, prices]
    )
    result.index.name = numeric_target.index.name or "delivery_start_utc"
    return result


def build_history_future_feature_matrix(
    historical_target: pd.Series,
    historical_covariates: pd.DataFrame,
    future_covariates: pd.DataFrame,
    *,
    price_lags: Sequence[int] = DEFAULT_PRICE_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    rolling_statistics: Sequence[str] = DEFAULT_ROLLING_STATISTICS,
    timezone: str = "Europe/Paris",
    scope: FeatureScope = "all",
) -> pd.DataFrame:
    """Combine known history and future PIT covariates without future prices.

    The public signature intentionally has no ``future_target`` argument.
    Future target rows are created as explicit ``NaN`` values before any lag
    or rolling operation.  Thus a 25-hour autumn delivery day cannot expose an
    earlier, already-delivered hour from that same forecast day via lag 24.

    Set ``scope="future"`` to return only forecast rows, or ``scope="all"``
    to receive the joint training/context and forecast feature matrix.
    """

    if scope not in {"all", "future"}:
        raise ValueError("scope doit être 'all' ou 'future'.")

    known_target = _numeric_target(
        historical_target,
        name="historical_target",
    )
    known_covariates = _validate_covariates(
        historical_covariates,
        expected_index=known_target.index,
        name="historical_covariates",
    )
    if not isinstance(future_covariates, pd.DataFrame):
        raise TypeError("future_covariates doit être un pandas.DataFrame.")
    future_index = validate_utc_hourly_index(
        future_covariates.index,
        name="future_covariates.index",
    )
    if future_covariates.empty:
        raise HourlyFeatureContractError("future_covariates est vide.")
    if not future_covariates.columns.is_unique:
        raise HourlyFeatureContractError(
            "future_covariates contient des colonnes dupliquées."
        )
    if set(future_covariates.columns) != set(known_covariates.columns):
        missing = sorted(
            set(known_covariates.columns) - set(future_covariates.columns)
        )
        unexpected = sorted(
            set(future_covariates.columns) - set(known_covariates.columns)
        )
        raise HourlyFeatureContractError(
            "Schéma de covariables historique/futur différent: "
            f"missing={missing}, unexpected={unexpected}."
        )
    expected_start = known_target.index[-1] + pd.Timedelta(hours=1)
    if future_index[0] != expected_start:
        raise HourlyFeatureContractError(
            "Le futur doit commencer exactement une heure après l'historique: "
            f"attendu={expected_start}, reçu={future_index[0]}."
        )

    ordered_future_covariates = future_covariates.loc[
        :, known_covariates.columns
    ].copy(deep=True)
    combined_covariates = pd.concat(
        [known_covariates, ordered_future_covariates],
        axis=0,
    )
    combined_target = pd.concat(
        [
            known_target,
            pd.Series(
                np.nan,
                index=future_index,
                dtype=float,
                name=known_target.name,
            ),
        ]
    )
    # Revalidate the boundary as well as each input segment.  No missing hour
    # is silently inserted between history and forecast.
    validate_utc_hourly_index(
        combined_target.index,
        name="combined.index",
    )
    features = build_hourly_feature_matrix(
        combined_target,
        combined_covariates,
        price_lags=price_lags,
        rolling_windows=rolling_windows,
        rolling_statistics=rolling_statistics,
        timezone=timezone,
    )
    if scope == "future":
        return features.loc[future_index].copy()
    return features


# A readable alias for orchestration code that treats the function as a join.
combine_history_and_future_features = build_history_future_feature_matrix


__all__ = [
    "DEFAULT_PRICE_LAGS",
    "DEFAULT_ROLLING_STATISTICS",
    "DEFAULT_ROLLING_WINDOWS",
    "HourlyFeatureContractError",
    "build_calendar_features",
    "build_history_future_feature_matrix",
    "build_hourly_feature_matrix",
    "combine_history_and_future_features",
    "validate_utc_hourly_index",
]
