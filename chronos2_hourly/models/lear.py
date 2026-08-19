"""Hourly LEAR-style ElasticNet forecaster.

LEAR is intentionally kept as a strong, transparent local baseline.  A robust
scaled ElasticNet predicts the conditional location.  Training-residual
quantiles, estimated separately for each sufficiently populated delivery hour,
turn that point forecast into q10/q50/q90 outputs.  The residual median also
aligns the point forecast with the MAE objective.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .base import (
    DEFAULT_QUANTILES,
    coerce_target,
    delivery_hours,
    make_prediction_frame,
    numeric_feature_frame,
    require_frame,
    validate_quantiles,
)


class HourlyLEAR:
    """One robust ElasticNet per local delivery hour, plus a global fallback."""

    def __init__(
        self,
        *,
        feature_columns: Sequence[str] | None = None,
        hour_column: str | None = "delivery_hour",
        timezone: str = "Europe/Paris",
        alpha: float = 0.02,
        l1_ratio: float = 0.8,
        max_iter: int = 10_000,
        min_samples_per_hour: int = 60,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        random_state: int = 42,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha doit être positif ou nul.")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio doit être compris entre 0 et 1.")
        if min_samples_per_hour < 2:
            raise ValueError("min_samples_per_hour doit être >= 2.")
        self.feature_columns = feature_columns
        self.hour_column = hour_column
        self.timezone = timezone
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.min_samples_per_hour = int(min_samples_per_hour)
        self.quantiles = validate_quantiles(quantiles)
        self.random_state = int(random_state)

    def _new_model(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "scale",
                    RobustScaler(quantile_range=(10.0, 90.0)),
                ),
                (
                    "elastic_net",
                    ElasticNet(
                        alpha=self.alpha,
                        l1_ratio=self.l1_ratio,
                        max_iter=self.max_iter,
                        selection="cyclic",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def _fit_one(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[Pipeline, np.ndarray]:
        model = self._new_model()
        model.fit(features, target.to_numpy())
        residuals = target.to_numpy() - np.asarray(model.predict(features), dtype=float)
        offsets = np.quantile(residuals, self.quantiles)
        return model, np.asarray(offsets, dtype=float)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | Sequence[float] | np.ndarray,
    ) -> "HourlyLEAR":
        """Fit using only the supplied training rows.

        Missing feature values are imputed from the corresponding training
        fold.  Rows with an unavailable target are never used.
        """

        require_frame(X)
        features, columns = numeric_feature_frame(X, self.feature_columns)
        target = coerce_target(y, X.index)
        hours = delivery_hours(
            X,
            hour_column=self.hour_column,
            timezone=self.timezone,
        )
        valid = target.notna().to_numpy()
        if valid.sum() < 2:
            raise ValueError("Pas assez de cibles observées pour entraîner LEAR.")

        self.feature_columns_ = columns
        self.global_model_, self.global_offsets_ = self._fit_one(
            features.loc[valid], target.loc[valid]
        )
        self.hour_models_: dict[int, Pipeline] = {}
        self.hour_offsets_: dict[int, np.ndarray] = {}
        for hour in sorted(set(hours[valid].tolist())):
            if hour < 0:
                continue
            mask = valid & (hours == hour)
            if int(mask.sum()) < self.min_samples_per_hour:
                continue
            model, offsets = self._fit_one(features.loc[mask], target.loc[mask])
            self.hour_models_[int(hour)] = model
            self.hour_offsets_[int(hour)] = offsets

        self.n_training_rows_ = int(valid.sum())
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return q10/q50/q90 forecasts with the exact index of ``X``."""

        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("HourlyLEAR doit être entraîné avant predict().")
        require_frame(X)
        features, _ = numeric_feature_frame(
            X,
            self.feature_columns,
            fitted_columns=self.feature_columns_,
        )
        hours = delivery_hours(
            X,
            hour_column=self.hour_column,
            timezone=self.timezone,
        )
        values = np.empty((len(X), len(self.quantiles)), dtype=float)
        for hour in np.unique(hours):
            mask = hours == hour
            model = self.hour_models_.get(int(hour), self.global_model_)
            offsets = self.hour_offsets_.get(int(hour), self.global_offsets_)
            point = np.asarray(model.predict(features.loc[mask]), dtype=float)
            values[mask, :] = point[:, None] + offsets[None, :]
        return make_prediction_frame(values, X.index, self.quantiles)

