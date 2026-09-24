"""Shared deterministic metrics and chronological split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index

from .config import SplitConfig


@dataclass(frozen=True)
class DaySplit:
    train_days: tuple[object, ...]
    validation_days: tuple[object, ...]
    test_days: tuple[object, ...]

    def phase_mask(self, index: pd.DatetimeIndex, *, timezone: str, phase: str) -> np.ndarray:
        days = {
            "train": self.train_days,
            "validation": self.validation_days,
            "test": self.test_days,
        }[phase]
        local = pd.Index(index.tz_convert(timezone).date)
        return np.asarray(local.isin(days), dtype=bool)


def validate_and_split_days(
    index: pd.DatetimeIndex,
    *,
    timezone: str,
    split: SplitConfig,
) -> DaySplit:
    if index.tz is None or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("La timeline doit etre UTC-aware, unique et croissante.")
    local_days = pd.Index(index.tz_convert(timezone).date)
    days = tuple(local_days.unique())
    needed = split.minimum_training_days + split.validation_days + split.test_days
    if len(days) < needed:
        raise ValueError(f"Historique insuffisant: {len(days)} jours < {needed}.")
    expected_days = tuple(pd.date_range(days[0], days[-1], freq="D").date)
    if days != expected_days:
        raise ValueError("La periode contient des jours locaux manquants.")
    for day in days:
        observed = index[local_days == day]
        expected = local_delivery_day_index(day, timezone=timezone)
        if not observed.equals(expected):
            raise ValueError(f"Journee locale incomplete: {day}.")
    test_start = len(days) - split.test_days
    validation_start = test_start - split.validation_days
    train = days[:validation_start]
    if len(train) < split.minimum_training_days:
        raise ValueError("La partition d'entrainement est trop courte.")
    return DaySplit(
        train_days=tuple(train),
        validation_days=tuple(days[validation_start:test_start]),
        test_days=tuple(days[test_start:]),
    )


def apply_horizon(
    frame: pd.DataFrame,
    *,
    timezone: str,
    horizon_hours: int | None,
) -> pd.DataFrame:
    if horizon_hours is None:
        return frame
    local_days = pd.Index(frame.index.tz_convert(timezone).date)
    position = pd.Series(np.arange(len(frame)), index=frame.index).groupby(local_days).cumcount()
    return frame.loc[position.to_numpy() < horizon_hours]


def point_metrics(actual: Iterable[float], prediction: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(prediction), dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("actual/prediction doivent etre des vecteurs finis et alignes.")
    error = p - y
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def quantile_metrics(
    actual: Iterable[float],
    q10: Iterable[float],
    q50: Iterable[float],
    q90: Iterable[float],
) -> dict[str, float]:
    y = np.asarray(list(actual), dtype=float)
    quantiles = {
        "q10": np.asarray(list(q10), dtype=float),
        "q50": np.asarray(list(q50), dtype=float),
        "q90": np.asarray(list(q90), dtype=float),
    }
    if any(values.shape != y.shape for values in quantiles.values()):
        raise ValueError("Les quantiles ne sont pas alignes avec actual.")
    if not np.isfinite(y).all() or not all(np.isfinite(values).all() for values in quantiles.values()):
        raise ValueError("Les quantiles et actual doivent etre finis.")
    if bool(((quantiles["q10"] > quantiles["q50"]) | (quantiles["q50"] > quantiles["q90"])).any()):
        raise ValueError("Croisement de quantiles detecte.")
    result = point_metrics(y, quantiles["q50"])
    for label, tau in (("q10", 0.1), ("q50", 0.5), ("q90", 0.9)):
        residual = y - quantiles[label]
        result[f"pinball_{label}"] = float(np.mean(np.maximum(tau * residual, (tau - 1.0) * residual)))
    result["coverage80"] = float(np.mean((y >= quantiles["q10"]) & (y <= quantiles["q90"])))
    result["interval_width80"] = float(np.mean(quantiles["q90"] - quantiles["q10"]))
    result["mean_observed_price"] = float(np.mean(y))
    result["mean_forecast_price"] = float(np.mean(quantiles["q50"]))
    result["mean_price_error"] = float(
        abs(result["mean_forecast_price"] - result["mean_observed_price"])
    )
    return result


def metrics_row(frame: pd.DataFrame, *, requested: Iterable[str]) -> dict[str, float | int]:
    required = {"actual", "q10", "q50", "q90"}
    if not required.issubset(frame):
        raise ValueError(f"Colonnes metriques absentes: {sorted(required.difference(frame))}.")
    values = quantile_metrics(frame["actual"], frame["q10"], frame["q50"], frame["q90"])
    return {"n_hours": int(len(frame)), **{name: values[name] for name in requested}}


def daily_metrics(frame: pd.DataFrame, *, timezone: str) -> pd.DataFrame:
    work = frame.copy()
    work["local_day"] = work.index.tz_convert(timezone).date
    rows: list[dict[str, object]] = []
    for day, block in work.groupby("local_day", sort=True):
        values = quantile_metrics(block["actual"], block["q10"], block["q50"], block["q90"])
        rows.append({"local_day": str(day), "n_hours": len(block), **values})
    return pd.DataFrame(rows)


__all__ = [
    "DaySplit",
    "apply_horizon",
    "daily_metrics",
    "metrics_row",
    "point_metrics",
    "quantile_metrics",
    "validate_and_split_days",
]
