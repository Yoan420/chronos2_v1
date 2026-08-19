"""Shared contracts for the supervised hourly forecasters.

The classes in this package deliberately use a small, sklearn-like contract:
``fit(X, y)`` consumes a feature :class:`pandas.DataFrame`, and ``predict(X)``
returns a DataFrame indexed like ``X`` with ``q10``, ``q50`` and ``q90``
columns.  Keeping this contract independent from the Chronos runner makes the
models easy to backtest and, importantly, easy to feed with genuinely OOF
Chronos predictions in the ensemble stage.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

import numpy as np
import pandas as pd


DEFAULT_QUANTILES: Final[tuple[float, ...]] = (0.1, 0.5, 0.9)


class HourlyModelError(RuntimeError):
    """Base exception for the hourly supervised models."""


class OptionalDependencyError(ImportError):
    """Raised when an explicitly requested optional model is unavailable."""


class LeakageRiskError(ValueError):
    """Raised when an operation cannot establish its OOF/causal contract."""


def quantile_column(quantile: float) -> str:
    """Return the stable column name used for a quantile."""

    value = float(quantile)
    if not 0.0 < value < 1.0:
        raise ValueError(f"Un quantile doit être dans ]0, 1[, reçu {value}.")
    percentage = value * 100.0
    if not np.isclose(percentage, round(percentage)):
        raise ValueError(
            "Les quantiles doivent être exprimables en pourcentage entier "
            f"pour nommer les colonnes, reçu {value}."
        )
    return f"q{int(round(percentage)):02d}"


def validate_quantiles(quantiles: Iterable[float]) -> tuple[float, ...]:
    """Validate, sort and de-duplicate a quantile collection."""

    values = tuple(sorted({float(value) for value in quantiles}))
    if not values:
        raise ValueError("Au moins un quantile est nécessaire.")
    for value in values:
        quantile_column(value)
    if 0.5 not in values:
        raise ValueError("Le quantile 0.5 est obligatoire pour optimiser la MAE.")
    return values


def require_frame(frame: pd.DataFrame, *, name: str = "X") -> pd.DataFrame:
    """Validate the public DataFrame input contract without mutating it."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} doit être un pandas.DataFrame.")
    if frame.empty:
        raise ValueError(f"{name} est vide.")
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Colonnes dupliquées dans {name}: {duplicates}.")
    return frame


def coerce_target(
    target: pd.Series | Sequence[float] | np.ndarray,
    index: pd.Index,
) -> pd.Series:
    """Convert a target to a positional numeric Series.

    Alignment is positional on purpose.  A local delivery index can contain two
    02:00 labels on the autumn DST day; an implicit ``reindex`` would be
    ambiguous in that perfectly legitimate case.
    """

    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise TypeError("y doit être unidimensionnel.")
        raw = target.iloc[:, 0].to_numpy(copy=False)
    elif isinstance(target, pd.Series):
        raw = target.to_numpy(copy=False)
    else:
        raw = np.asarray(target)
    if raw.ndim != 1:
        raise TypeError("y doit être unidimensionnel.")
    if len(raw) != len(index):
        raise ValueError(
            f"X et y n'ont pas la même longueur: {len(index)} != {len(raw)}."
        )
    numeric = pd.to_numeric(pd.Series(raw, index=index), errors="coerce")
    return numeric.astype(float)


def numeric_feature_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None,
    *,
    fitted_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Select a stable, finite numeric feature schema.

    Infinite values are converted to missing values and are subsequently
    handled by each model's training-only imputer.  Entirely missing columns
    are rejected rather than silently changing the schema between folds.
    """

    require_frame(frame)
    if fitted_columns is not None:
        columns = list(fitted_columns)
    elif feature_columns is not None:
        columns = list(feature_columns)
    else:
        columns = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if not columns:
        raise ValueError("Aucune feature numérique n'a été fournie.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Features absentes du DataFrame: {missing}.")

    work = frame.loc[:, columns].copy()
    invalid_types = [
        column
        for column in columns
        if not (
            pd.api.types.is_numeric_dtype(work[column])
            or pd.api.types.is_bool_dtype(work[column])
        )
    ]
    if invalid_types:
        raise TypeError(
            "Les modèles horaires attendent des features numériques; encodez "
            f"d'abord les colonnes {invalid_types}."
        )
    work = work.astype(float).replace([np.inf, -np.inf], np.nan)
    if fitted_columns is None:
        empty = [column for column in columns if work[column].notna().sum() == 0]
        if empty:
            raise ValueError(
                "Features entièrement manquantes dans l'échantillon "
                f"d'entraînement: {empty}."
            )
    return work, columns


def delivery_hours(
    frame: pd.DataFrame,
    *,
    hour_column: str | None,
    timezone: str,
) -> np.ndarray:
    """Extract local delivery hours while preserving 23/25-hour DST days."""

    if hour_column and hour_column in frame.columns:
        values = pd.to_numeric(frame[hour_column], errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{hour_column} contient des valeurs manquantes.")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{hour_column} doit contenir des heures entières.")
        hours = values.astype(np.int16)
    elif isinstance(frame.index, pd.DatetimeIndex):
        index = frame.index
        if index.tz is not None:
            index = index.tz_convert(timezone)
        hours = index.hour.to_numpy(dtype=np.int16)
    else:
        # A global model remains available for feature matrices without a
        # delivery timestamp.  Per-hour models require either input above.
        return np.full(len(frame), -1, dtype=np.int16)
    if ((hours < 0) | (hours > 23)).any():
        raise ValueError("Les heures de livraison doivent être comprises entre 0 et 23.")
    return hours


def make_prediction_frame(
    values: np.ndarray,
    index: pd.Index,
    quantiles: Sequence[float],
    *,
    enforce_non_crossing: bool = True,
) -> pd.DataFrame:
    """Build the common quantile output and optionally repair crossings."""

    array = np.asarray(values, dtype=float)
    if array.shape != (len(index), len(quantiles)):
        raise ValueError(
            "Forme de prédiction inattendue: "
            f"{array.shape}, attendu {(len(index), len(quantiles))}."
        )
    if enforce_non_crossing:
        # Keep the dedicated median forecast untouched: sorting all values can
        # silently replace q50 with an estimate trained for another quantile
        # and therefore change the point forecast scored with MAE.  Repair the
        # two tails monotonically outwards from the median instead.
        array = array.copy()
        median_position = tuple(float(value) for value in quantiles).index(0.5)
        for position in range(median_position - 1, -1, -1):
            array[:, position] = np.minimum(
                array[:, position], array[:, position + 1]
            )
        for position in range(median_position + 1, len(quantiles)):
            array[:, position] = np.maximum(
                array[:, position], array[:, position - 1]
            )
    return pd.DataFrame(
        array,
        index=index,
        columns=[quantile_column(value) for value in quantiles],
    )
