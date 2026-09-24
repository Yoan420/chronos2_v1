"""Isolated, loss-only CatBoost residual experiment; no I/O or activation.

The operational residual class, builder, and configuration files are never
modified. Each requested day is fitted independently on the previous 365
complete local civil days, using the frozen Chronos forecast as its base.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import inspect
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_hourly.models.residual_corrector import (
    ResidualCorrector,
    ResidualMetaFeatureBuilder,
)
from chronos2_hourly.nuclear_forecast import _validate_origins


QUANTILES = ("q10", "q50", "q90")
TRAINING_DAYS = 365
FIXED_RECIPE = {
    "backend": "catboost", "iterations": 700, "depth": 6,
    "learning_rate": 0.03, "l2_leaf_reg": 15.0, "random_state": 42,
    "min_training_rows": 720, "min_samples_leaf": 30,
    "correction_scale": 1.0, "max_abs_correction": 40.0,
    "correction_clip": None,
}


class RMSEExperimentError(ValueError):
    """The experimental replay does not satisfy the frozen causal contract."""


class LossOnlyResidualCorrector(ResidualCorrector):
    """Keep all inherited behavior; override only CatBoost loss and metric."""

    experiment_loss = "RMSE"

    def _new_model(self) -> Any:
        if self.backend_ != "catboost":
            raise RMSEExperimentError("This experiment requires CatBoost; no fallback is allowed.")
        return super()._new_model().set_params(
            loss_function=self.experiment_loss, eval_metric=self.experiment_loss,
        )


def _constructor_kwargs(prototype: ResidualCorrector) -> dict[str, Any]:
    return {
        name: deepcopy(getattr(prototype, name))
        for name, parameter in inspect.signature(ResidualCorrector.__init__).parameters.items()
        if name != "self" and parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
        )
    }


def make_corrector(
    config: Mapping[str, Any], timezone: str, threads: int, loss: str = "RMSE",
) -> LossOnlyResidualCorrector:
    """Clone the existing configured recipe without modifying its configuration."""
    from run_chronos2_hourly import _residual_corrector_factory

    if loss not in {"MAE", "RMSE"}:
        raise RMSEExperimentError("loss must be MAE or RMSE.")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise RMSEExperimentError("threads must be a positive integer.")
    factory, base_model = _residual_corrector_factory(deepcopy(dict(config)), timezone=timezone)
    if factory is None or base_model != "chronos2":
        raise RMSEExperimentError("An enabled Chronos-based residual recipe is required.")
    prototype = factory()
    kwargs = _constructor_kwargs(prototype)
    mismatches = {key: kwargs[key] for key, value in FIXED_RECIPE.items() if kwargs[key] != value}
    if mismatches:
        raise RMSEExperimentError(f"Frozen residual recipe mismatch: {mismatches}.")
    builder = prototype.feature_builder or ResidualMetaFeatureBuilder(**prototype.feature_builder_options)
    if not builder.exclude_historical_prices:
        raise RMSEExperimentError("Historical observed-price features must remain excluded.")
    if builder.timezone != timezone:
        raise RMSEExperimentError("Feature-builder timezone differs from the delivery timezone.")
    kwargs["thread_count"] = threads
    corrector = LossOnlyResidualCorrector(**kwargs)
    corrector.experiment_loss = loss
    # Resolve now, before any fit: absent CatBoost must not silently substitute sklearn.
    if corrector._resolve_backend() != "catboost":
        raise RMSEExperimentError("CatBoost is required.")
    return corrector


def _frame(value: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise RMSEExperimentError(f"{name} must be a DataFrame.")
    index = value.index
    if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC"
            or index.hasnans or not index.is_unique or not index.is_monotonic_increasing
            or not index.equals(index.floor("h")) or not value.columns.is_unique):
        raise RMSEExperimentError(f"{name} requires unique, sorted, hourly UTC rows and unique columns.")
    return value


def _civil_index(start: date, end_exclusive: date, timezone: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start, tz=timezone),
                         pd.Timestamp(end_exclusive, tz=timezone),
                         freq="h", inclusive="left").tz_convert("UTC")


def _base(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not set(QUANTILES).issubset(frame):
        raise RMSEExperimentError(f"{name} requires q10/q50/q90.")
    result = frame.loc[:, list(QUANTILES)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise RMSEExperimentError(f"{name} quantiles must be finite.")
    if ((result.q10 > result.q50) | (result.q50 > result.q90)).any():
        raise RMSEExperimentError(f"{name} quantiles cross.")
    return result


def fit_day(
    *, history: pd.DataFrame, future: pd.DataFrame, features: pd.DataFrame,
    timezone: str, day: date | str, config: Mapping[str, Any], threads: int = 2,
    loss: str = "RMSE", expected_features: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit/predict one day; only training-window labels are ever accessed.

    ``future`` is exactly the requested day, including when replaying an old
    day from ``history``. It must not carry an observed target. The caller may
    join evaluation labels to the returned predictions only after this call.
    ``expected_features`` freezes the exact ordered inherited meta-schema.
    """
    started = perf_counter()
    parsed = pd.Timestamp(day)
    if pd.isna(parsed) or parsed.tzinfo is not None or parsed != parsed.normalize():
        raise RMSEExperimentError("day must be an explicit local civil date.")
    target_day = parsed.date()
    history = _frame(history, "history")
    future = _frame(future, "future")
    features = _frame(features, "features")
    training_start = target_day - timedelta(days=TRAINING_DAYS)
    training_index = _civil_index(training_start, target_day, timezone)
    prediction_index = _civil_index(target_day, target_day + timedelta(days=1), timezone)
    if not future.index.equals(prediction_index):
        raise RMSEExperimentError("future must be the complete physical prediction day (23/24/25 hours).")
    if "actual" in future and future.actual.notna().any():
        raise RMSEExperimentError("Prediction-day actual labels are forbidden in future.")
    if len(training_index.difference(history.index)):
        raise RMSEExperimentError("Training requires all previous 365 complete civil days, D-365..D-1.")
    selected_index = training_index.append(prediction_index)
    if len(selected_index.difference(features.index)):
        raise RMSEExperimentError("features do not cover every training and prediction hour.")
    training = history.loc[training_index]
    _validate_origins(training, timezone)
    _validate_origins(future, timezone)
    train_base, prediction_base = _base(training, "training"), _base(future, "future")
    if "actual" not in training:
        raise RMSEExperimentError("Training labels actual are required.")
    actual = pd.to_numeric(training.actual, errors="coerce")
    if not np.isfinite(actual.to_numpy(dtype=float)).all():
        raise RMSEExperimentError("Training labels must be finite.")
    train_experts = train_base.rename(columns={q: f"chronos2__{q}" for q in QUANTILES})
    prediction_experts = prediction_base.rename(columns={q: f"chronos2__{q}" for q in QUANTILES})
    corrector = make_corrector(config, timezone, threads, loss)
    fit_started = perf_counter()
    corrector.fit(features.loc[training_index], actual, train_base, train_experts)
    fit_seconds = perf_counter() - fit_started
    columns = tuple(map(str, corrector.feature_columns_))
    if expected_features is not None and tuple(expected_features) != columns:
        raise RMSEExperimentError("Inherited residual feature schema differs from the frozen baseline.")
    predicted = corrector.predict(features.loc[prediction_index], prediction_base, prediction_experts)
    predicted = _base(_frame(predicted, "predicted"), "predicted")
    if not predicted.index.equals(prediction_index):
        raise RMSEExperimentError("Predictions do not preserve the physical delivery index.")
    correction = corrector.predict_correction(features.loc[prediction_index], prediction_base, prediction_experts)
    raw = correction.attrs.get("raw_correction")
    if (not isinstance(raw, pd.Series) or not raw.index.equals(prediction_index)
            or not correction.index.equals(prediction_index)
            or not np.isfinite(raw.to_numpy(dtype=float)).all()
            or not np.isfinite(correction.to_numpy(dtype=float)).all()):
        raise RMSEExperimentError("Finite aligned raw/applied correction diagnostics are required.")
    expected_shift = np.clip(raw.to_numpy(dtype=float), -40.0, 40.0)
    if not np.allclose(correction.to_numpy(dtype=float), expected_shift, rtol=0, atol=1e-10):
        raise RMSEExperimentError("Applied correction differs from the frozen +/-40 cap.")
    differences = predicted.to_numpy(dtype=float) - prediction_base.to_numpy(dtype=float)
    if not np.allclose(differences, expected_shift[:, None], rtol=0, atol=1e-8):
        raise RMSEExperimentError("All quantiles must receive exactly the same capped residual shift.")
    output = predicted.copy()
    output["raw_correction"] = raw.to_numpy(dtype=float)
    output["applied_correction"] = correction.to_numpy(dtype=float)
    output["forecast_origin_utc"] = future.forecast_origin_utc
    output.index.name = "delivery_start_utc"
    # The inherited prediction stores a Series in attrs; keep tabular artifacts
    # serializable and place all experiment provenance in the separate audit.
    output.attrs.clear()
    constructor = _constructor_kwargs(corrector)
    constructor.pop("feature_builder")
    audit = {
        "schema_version": 1, "delivery_day": target_day.isoformat(), "timezone": timezone,
        "method": "independent_daily_catboost_residual_rolling365", "loss": loss,
        "eval_metric": loss, "constructor_parameters": constructor,
        "catboost_parameters": corrector.model_.get_params(),
        "parameter_parity_scope": "explicit_parameters_except_loss_and_eval_metric",
        "objective_specific_catboost_defaults_may_differ": True,
        "training_days": TRAINING_DAYS, "training_rows": len(training_index),
        "training_start_day": training_start.isoformat(),
        "training_end_day": (target_day - timedelta(days=1)).isoformat(),
        "training_first_utc": training_index[0].isoformat(),
        "training_last_utc": training_index[-1].isoformat(),
        "forecast_origin_utc": pd.Timestamp(future.forecast_origin_utc.iloc[0]).isoformat(),
        "forecast_hours": len(prediction_index), "feature_columns": list(columns),
        "n_features": len(columns), "fit_seconds": fit_seconds,
        "elapsed_seconds": perf_counter() - started,
        "clipped_hours": int((np.abs(raw.to_numpy(dtype=float)) > 40.0).sum()),
        "maximum_absolute_raw_correction": float(np.abs(raw.to_numpy(dtype=float)).max()),
        "maximum_absolute_applied_correction": float(np.abs(correction.to_numpy(dtype=float)).max()),
        "current_day_labels_used": False, "causality_violations": 0,
        "quantile_shift_invariant": True, "chronos_recomputed": False,
        "kalman_used": False, "production_modified": False,
    }
    return output, audit


__all__ = ["FIXED_RECIPE", "LossOnlyResidualCorrector", "RMSEExperimentError", "fit_day", "make_corrector"]
