"""Hourly nonlinear quantile model with an optional CatBoost backend."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .base import (
    DEFAULT_QUANTILES,
    OptionalDependencyError,
    coerce_target,
    delivery_hours,
    make_prediction_frame,
    numeric_feature_frame,
    require_frame,
    validate_quantiles,
)


def catboost_available() -> bool:
    """Return whether CatBoost can be imported without importing it eagerly."""

    return importlib.util.find_spec("catboost") is not None


class HourlyCatBoost:
    """Quantile regressors by local hour with a safe sklearn fallback.

    ``backend='auto'`` uses CatBoost when installed and otherwise falls back to
    sklearn's histogram gradient boosting.  ``backend='catboost'`` is strict
    and raises an actionable error when the optional dependency is missing.
    The explicit ``'sklearn'`` backend keeps CI and CPU-only deployments fully
    testable.
    """

    def __init__(
        self,
        *,
        feature_columns: Sequence[str] | None = None,
        hour_column: str | None = "delivery_hour",
        timezone: str = "Europe/Paris",
        backend: str = "auto",
        min_samples_per_hour: int = 120,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        iterations: int = 700,
        depth: int = 7,
        learning_rate: float = 0.035,
        l2_leaf_reg: float = 6.0,
        min_samples_leaf: int = 30,
        random_state: int = 42,
        thread_count: int = -1,
        verbose: bool | int = False,
    ) -> None:
        if backend not in {"auto", "catboost", "sklearn"}:
            raise ValueError("backend doit valoir auto, catboost ou sklearn.")
        if min_samples_per_hour < 2:
            raise ValueError("min_samples_per_hour doit être >= 2.")
        self.feature_columns = feature_columns
        self.hour_column = hour_column
        self.timezone = timezone
        self.backend = backend
        self.min_samples_per_hour = int(min_samples_per_hour)
        self.quantiles = validate_quantiles(quantiles)
        self.iterations = int(iterations)
        self.depth = int(depth)
        self.learning_rate = float(learning_rate)
        self.l2_leaf_reg = float(l2_leaf_reg)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self.thread_count = int(thread_count)
        self.verbose = verbose

    def _resolve_backend(self) -> str:
        available = catboost_available()
        if self.backend == "catboost" and not available:
            raise OptionalDependencyError(
                "CatBoost a été demandé mais n'est pas installé. Exécutez "
                "`python -m pip install catboost>=1.2` ou utilisez "
                "backend='sklearn'."
            )
        if self.backend == "auto":
            return "catboost" if available else "sklearn"
        return self.backend

    def _new_model(self, quantile: float) -> Any:
        if self.backend_ == "catboost":
            # Lazy import keeps the whole package importable without CatBoost.
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                loss_function=f"Quantile:alpha={quantile}",
                eval_metric=f"Quantile:alpha={quantile}",
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                random_seed=self.random_state,
                thread_count=self.thread_count,
                allow_writing_files=False,
                verbose=self.verbose,
                has_time=True,
                nan_mode="Min",
            )
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            learning_rate=self.learning_rate,
            max_iter=self.iterations,
            max_leaf_nodes=max(3, 2**self.depth - 1),
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_leaf_reg,
            random_state=self.random_state,
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )

    def _fit_models(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> dict[float, Any]:
        models: dict[float, Any] = {}
        for quantile in self.quantiles:
            model = self._new_model(quantile)
            model.fit(features, target.to_numpy())
            models[quantile] = model
        return models

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | Sequence[float] | np.ndarray,
    ) -> "HourlyCatBoost":
        require_frame(X)
        self.backend_ = self._resolve_backend()
        features, columns = numeric_feature_frame(X, self.feature_columns)
        target = coerce_target(y, X.index)
        hours = delivery_hours(
            X,
            hour_column=self.hour_column,
            timezone=self.timezone,
        )
        valid = target.notna().to_numpy()
        if valid.sum() < 2:
            raise ValueError("Pas assez de cibles observées pour entraîner le modèle.")

        self.feature_columns_ = columns
        self.global_models_ = self._fit_models(
            features.loc[valid], target.loc[valid]
        )
        self.hour_models_: dict[int, dict[float, Any]] = {}
        for hour in sorted(set(hours[valid].tolist())):
            if hour < 0:
                continue
            mask = valid & (hours == hour)
            if int(mask.sum()) < self.min_samples_per_hour:
                continue
            self.hour_models_[int(hour)] = self._fit_models(
                features.loc[mask], target.loc[mask]
            )
        self.n_training_rows_ = int(valid.sum())
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("HourlyCatBoost doit être entraîné avant predict().")
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
            models = self.hour_models_.get(int(hour), self.global_models_)
            for column, quantile in enumerate(self.quantiles):
                values[mask, column] = np.asarray(
                    models[quantile].predict(features.loc[mask]), dtype=float
                )
        # Independent quantile models may cross.  Sorting is deterministic and
        # preserves the median column used by MAE scoring.
        return make_prediction_frame(values, X.index, self.quantiles)

