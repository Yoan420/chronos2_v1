"""Chronological conditional-mean price experts for the RMSE objective.

This is a separate recipe from the frozen MAE models. Neither the residual
target nor its predicted correction is clipped. Storm never enters the model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

PROTOCOL = "nyx_squared_error_price_experts_v1"


@dataclass(frozen=True)
class PriceModelConfig:
    target_mode: str = "residual"
    iterations: int = 400
    depth: int = 6
    learning_rate: float = .035
    l2_leaf_reg: float = 20.
    seed: int = 20260925


def frame_hash(frame):
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    names = list(frame.columns) if isinstance(frame, pd.DataFrame) else [frame.name]
    return hashlib.sha256(values + json.dumps(names).encode()).hexdigest()


def grid(first, stop, timezone="Europe/Paris"):
    return pd.date_range(pd.Timestamp(first, tz=timezone), pd.Timestamp(stop, tz=timezone),
                         freq="h", inclusive="left").tz_convert("UTC")


def parameters(config):
    if not isinstance(config, PriceModelConfig) or config.target_mode not in ("residual", "absolute"):
        raise ValueError("Explicit residual or absolute squared-error recipe required")
    if (isinstance(config.iterations, bool) or not isinstance(config.iterations, int)
            or not 1 <= config.iterations <= 2000 or isinstance(config.depth, bool)
            or not isinstance(config.depth, int) or not 1 <= config.depth <= 10
            or not np.isfinite(config.learning_rate) or not 0 < config.learning_rate <= 1
            or not np.isfinite(config.l2_leaf_reg) or config.l2_leaf_reg < 0):
        raise ValueError("Invalid squared-error configuration")
    return {"loss": "squared_error", "learning_rate": config.learning_rate,
            "max_iter": config.iterations, "max_leaf_nodes": 31, "max_depth": config.depth,
            "min_samples_leaf": 30, "l2_regularization": config.l2_leaf_reg,
            "max_bins": 127, "early_stopping": False, "random_state": config.seed}


def fit_price_block(features, actual, nyx_point, *, origin_day, stop_day,
                    config=PriceModelConfig(), timezone="Europe/Paris",
                    initial_training_day="2024-09-23"):
    started = time.monotonic()
    settings = parameters(config)
    origin, stop, initial = (pd.Timestamp(value).date() for value in (origin_day, stop_day, initial_training_day))
    if not origin < stop <= origin + timedelta(days=7):
        raise ValueError("One to seven complete prediction days required")
    if not isinstance(features, pd.DataFrame) or features.empty or not features.columns.is_unique:
        raise ValueError("Unique nonempty feature columns required")
    if any(any(word in str(name).lower() for word in ("storm", "actual", "target", "observed"))
           for name in features.columns):
        raise ValueError("Storm and observation columns are forbidden features")
    index = features.index
    if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC" or index.hasnans
            or not index.is_unique or not index.is_monotonic_increasing or not index.equals(index.floor("h"))):
        raise ValueError("Unique ordered physical UTC hourly features required")
    for value in (actual, nyx_point):
        if not isinstance(value, pd.Series) or not value.index.equals(index) or "storm" in str(value.name).lower():
            raise ValueError("Aligned actual and NYX series required; Storm forbidden")
    first = max(initial, origin - timedelta(days=365))
    training_days = (origin - first).days
    if training_days < 90 or (origin >= pd.Timestamp("2025-09-24").date() and training_days != 365):
        raise ValueError("At least 90 initial days and exactly 365 past days for scored origins required")
    train_index, current_index = grid(first, origin, timezone), grid(origin, stop, timezone)
    if not train_index.isin(index).all() or not current_index.isin(index).all():
        raise ValueError("Incomplete training or prediction civil grid")
    train_x, current_x = features.loc[train_index].astype(float), features.loc[current_index].astype(float)
    if np.isinf(train_x.to_numpy()).any() or np.isinf(current_x.to_numpy()).any():
        raise ValueError("Infinite feature")
    train_y = actual.loc[train_index].astype(float)
    train_base, current_base = nyx_point.loc[train_index].astype(float), nyx_point.loc[current_index].astype(float)
    if not all(np.isfinite(value.to_numpy()).all() for value in (train_y, train_base, current_base)):
        raise ValueError("Finite past labels and NYX forecasts required")
    target = train_y - train_base if config.target_mode == "residual" else train_y
    with threadpool_limits(limits=1):
        estimator = HistGradientBoostingRegressor(**settings)
        estimator.fit(train_x, target)
        prediction = estimator.predict(current_x)
    if config.target_mode == "residual":
        prediction = current_base.to_numpy() + prediction
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite squared-error prediction")
    audit = {"protocol": PROTOCOL, "origin_day": str(origin), "stop_day_exclusive": str(stop),
             "training_first_day": str(first), "training_last_day": str(origin-timedelta(days=1)),
             "training_days": training_days, "training_hours": len(train_index), "prediction_hours": len(current_index),
             "training_labels_before_origin": True, "daily_features_keep_their_own_D_minus_1_08_origin": True,
             "training_features_sha256": frame_hash(train_x), "training_target_sha256": frame_hash(target),
             "prediction_features_sha256": frame_hash(current_x), "config": asdict(config), "parameters": settings,
             "target_clipped": False, "prediction_clipped": False, "Storm_used_as_input": False,
             "fit_seconds": time.monotonic()-started}
    return pd.DataFrame({"point": prediction}, index=current_index), audit
