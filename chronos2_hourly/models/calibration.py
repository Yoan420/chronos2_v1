"""Strictly causal rolling-median calibration for hourly price forecasts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .base import require_frame


def _timestamps(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(
            "La calibration causale exige un DatetimeIndex de livraison."
        )
    index = frame.index
    if index.hasnans:
        raise ValueError("L'index de livraison contient des timestamps manquants.")
    if index.tz is None and index.duplicated().any():
        raise ValueError(
            "Un index local naïf dupliqué est ambigu au changement d'heure; "
            "utilisez des timestamps UTC ou timezone-aware."
        )
    return index


class RollingMedianCalibrator:
    """Add the median of *past* observed residuals to every quantile.

    Residuals are defined as ``actual - q50``.  ``fit_transform`` computes each
    correction using rows strictly earlier than the row being calibrated; it
    is therefore suitable for OOF backtests.  ``fit`` + ``transform`` applies
    the same rule by searching the stored history strictly before every future
    delivery timestamp.
    """

    def __init__(
        self,
        *,
        window: int = 24 * 28,
        min_periods: int = 24,
        by_delivery_hour: bool = True,
        timezone: str = "Europe/Paris",
        max_abs_correction: float | None = 50.0,
    ) -> None:
        if window < 1 or min_periods < 1 or min_periods > window:
            raise ValueError("Il faut 1 <= min_periods <= window.")
        if max_abs_correction is not None and max_abs_correction <= 0:
            raise ValueError("max_abs_correction doit être positif.")
        self.window = int(window)
        self.min_periods = int(min_periods)
        self.by_delivery_hour = bool(by_delivery_hour)
        self.timezone = timezone
        self.max_abs_correction = max_abs_correction

    def _validate_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        require_frame(predictions, name="predictions")
        missing = [column for column in ("q10", "q50", "q90") if column not in predictions]
        if missing:
            raise ValueError(f"Quantiles absents des prédictions: {missing}.")
        work = predictions.loc[:, ["q10", "q50", "q90"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(work.to_numpy()).all():
            raise ValueError("Les prédictions à calibrer doivent être finies.")
        _timestamps(work)
        return work

    def _local_hours(self, index: pd.DatetimeIndex) -> np.ndarray:
        local = index.tz_convert(self.timezone) if index.tz is not None else index
        return local.hour.to_numpy(dtype=np.int16)

    def _clip(self, correction: float) -> float:
        if self.max_abs_correction is None or not np.isfinite(correction):
            return correction
        return float(
            np.clip(
                correction,
                -self.max_abs_correction,
                self.max_abs_correction,
            )
        )

    def fit(
        self,
        predictions: pd.DataFrame,
        y: pd.Series | Sequence[float] | np.ndarray,
    ) -> "RollingMedianCalibrator":
        work = self._validate_predictions(predictions)
        target = pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce").to_numpy(
            dtype=float
        )
        if len(target) != len(work):
            raise ValueError("predictions et y n'ont pas la même longueur.")
        residual = target - work["q50"].to_numpy(dtype=float)
        valid = np.isfinite(residual)
        if not valid.any():
            raise ValueError("Aucun résidu observé pour calibrer les prévisions.")
        index = _timestamps(work)
        # Store UTC nanoseconds to compare timezone-aware timestamps safely.
        if index.tz is not None:
            ordered_ns = index.tz_convert("UTC").asi8
        else:
            ordered_ns = index.asi8
        order = np.argsort(ordered_ns, kind="stable")
        hours = self._local_hours(index)
        history = pd.DataFrame(
            {
                "timestamp_ns": ordered_ns[order],
                "hour": hours[order],
                "residual": residual[order],
            }
        )
        history = history.loc[np.isfinite(history["residual"])].reset_index(drop=True)
        self.history_ = history
        self.is_fitted_ = True
        return self

    def _history_correction(self, timestamp_ns: int, hour: int) -> float:
        history = self.history_
        prior = history.loc[history["timestamp_ns"] < timestamp_ns]
        if self.by_delivery_hour:
            hourly = prior.loc[prior["hour"] == hour, "residual"].tail(self.window)
            if len(hourly) >= self.min_periods:
                return self._clip(float(hourly.median()))
        global_values = prior["residual"].tail(self.window)
        if len(global_values) >= self.min_periods:
            return self._clip(float(global_values.median()))
        return 0.0

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("RollingMedianCalibrator doit être entraîné.")
        work = self._validate_predictions(predictions)
        index = _timestamps(work)
        query_ns = index.tz_convert("UTC").asi8 if index.tz is not None else index.asi8
        hours = self._local_hours(index)
        correction = np.asarray(
            [
                self._history_correction(int(timestamp_ns), int(hour))
                for timestamp_ns, hour in zip(query_ns, hours, strict=True)
            ],
            dtype=float,
        )
        result = work.add(correction, axis=0)
        result.attrs["median_correction"] = pd.Series(correction, index=work.index)
        return result

    def fit_transform(
        self,
        predictions: pd.DataFrame,
        y: pd.Series | Sequence[float] | np.ndarray,
    ) -> pd.DataFrame:
        """Causally calibrate an OOF history, excluding each current residual."""

        self.fit(predictions, y)
        return self.transform(predictions)

