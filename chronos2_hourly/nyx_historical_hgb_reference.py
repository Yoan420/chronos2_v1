"""CPU implementation of the three HGB experts in the annual CWE reference.

The archived 2026 study fitted each expert weekly on the preceding 365 civil
days (with a shorter warmup before 2025-09-24). Feature values are supplied by
the caller: this module does not claim to collect future Saturn/NYX/Test2 data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


TIMEZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin",
             "BE": "Europe/Brussels", "NL": "Europe/Amsterdam"}
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "config" / "nyx_historical_hgb_feature_schemas.json"
PARAMETERS = {
    "loss": "absolute_error", "learning_rate": .035, "max_iter": 400,
    "max_leaf_nodes": 31, "max_depth": 6, "min_samples_leaf": 30,
    "l2_regularization": 20., "max_bins": 127,
    "early_stopping": False, "random_state": 20260925,
}
VARIANTS = {
    "hist_residual_400": ("base", "residual"),
    "hist_absolute_400": ("base", "absolute"),
    "augmented_hist_residual_400": ("augmented", "residual"),
}


@dataclass(frozen=True)
class HGBFit:
    model: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...]
    variant: str
    zone: str
    origin_day: str
    stop_day_exclusive: str


def frame_hash(frame: pd.DataFrame | pd.Series) -> str:
    """Reproduce the archived model's pandas value/index/ordered-name digest."""
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    names = list(frame.columns) if isinstance(frame, pd.DataFrame) else [frame.name]
    return hashlib.sha256(values + json.dumps(names).encode()).hexdigest()


def civil_grid(first: str, stop: str, timezone: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(first, tz=timezone), pd.Timestamp(stop, tz=timezone),
                         freq="h", inclusive="left").tz_convert("UTC")


def archived_columns(zone: str, variant: str, path: Path = SCHEMA_PATH) -> tuple[str, ...]:
    if zone not in TIMEZONES or variant not in VARIANTS:
        raise ValueError("Unknown country or archived HGB variant")
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("protocol") != "nyx_historical_hgb_feature_schemas_v1":
        raise ValueError("Unknown feature schema protocol")
    country = document["countries"][zone]
    names = tuple(country["base"])
    if VARIANTS[variant][0] == "augmented":
        names += tuple(country["augmented_extra"])
    expected = 334 if VARIANTS[variant][0] == "augmented" else 292
    if len(names) != expected or len(set(names)) != len(names):
        raise ValueError("Incomplete archived HGB feature schema")
    return names


def _check_index(index: pd.Index) -> None:
    if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC"
            or index.hasnans or not index.is_unique or not index.is_monotonic_increasing
            or not index.equals(index.floor("h"))):
        raise ValueError("Ordered, unique hourly UTC features required")


def fit_hgb_block(
    features: pd.DataFrame, actual: pd.Series, nyx_point: pd.Series, *,
    zone: str, variant: str, origin_day: str, stop_day: str,
    expected_columns: tuple[str, ...] | None = None,
    initial_training_day: str = "2024-09-23",
) -> tuple[HGBFit, pd.DataFrame, dict]:
    """Fit one historical HGB expert and predict 1–7 complete civil days.

    ``actual`` may be missing for the prediction interval. Its training rows
    must be present and finite. Feature rows for both intervals must use the
    archived ordered schema. A frozen weekly fit may be reused for its other
    delivery days only when their feature schema and origin are unchanged.
    """
    if zone not in TIMEZONES or variant not in VARIANTS:
        raise ValueError("Unknown country or archived HGB variant")
    origin, stop = pd.Timestamp(origin_day).date(), pd.Timestamp(stop_day).date()
    initial = pd.Timestamp(initial_training_day).date()
    if not origin < stop <= origin + timedelta(days=7):
        raise ValueError("One to seven complete prediction days required")
    first = max(initial, origin - timedelta(days=365))
    if (origin - first).days < 90:
        raise ValueError("At least 90 initial training days required")
    if origin >= pd.Timestamp("2025-09-24").date() and (origin - first).days != 365:
        raise ValueError("Scored origins require 365 training days")
    if not isinstance(features, pd.DataFrame) or features.empty or not features.columns.is_unique:
        raise ValueError("Unique nonempty feature columns required")
    _check_index(features.index)
    if expected_columns is None:
        expected_columns = archived_columns(zone, variant)
    if tuple(features.columns) != tuple(expected_columns):
        raise ValueError("Archived HGB feature order/schema changed")
    if any(any(word in str(name).lower() for word in ("storm", "actual", "target", "observed"))
           for name in features.columns):
        raise ValueError("Observation or Storm column in features")
    for series in (actual, nyx_point):
        if (not isinstance(series, pd.Series) or not series.index.equals(features.index)
                or "storm" in str(series.name).lower()):
            raise ValueError("Aligned actual and NYX series required; Storm forbidden")
    train_index = civil_grid(str(first), str(origin), TIMEZONES[zone])
    current_index = civil_grid(str(origin), str(stop), TIMEZONES[zone])
    if not train_index.isin(features.index).all() or not current_index.isin(features.index).all():
        raise ValueError("Incomplete training or prediction civil grid")
    train_x = features.loc[train_index].astype(float)
    current_x = features.loc[current_index].astype(float)
    if np.isinf(train_x.to_numpy()).any() or np.isinf(current_x.to_numpy()).any():
        raise ValueError("Infinite HGB feature")
    train_y = actual.loc[train_index].astype(float)
    train_base = nyx_point.loc[train_index].astype(float)
    current_base = nyx_point.loc[current_index].astype(float)
    if not all(np.isfinite(s.to_numpy()).all() for s in (train_y, train_base, current_base)):
        raise ValueError("Nonfinite training label or NYX point")
    target_mode = VARIANTS[variant][1]
    target = train_y - train_base if target_mode == "residual" else train_y
    model = HistGradientBoostingRegressor(**PARAMETERS)
    with threadpool_limits(limits=1):
        model.fit(train_x, target)
        raw = model.predict(current_x)
    point = current_base.to_numpy() + np.clip(raw, -60., 60.) if target_mode == "residual" else raw
    if not np.isfinite(point).all():
        raise ValueError("Nonfinite HGB point")
    fitted = HGBFit(model, tuple(features.columns), variant, zone, str(origin), str(stop))
    audit = {
        "variant": variant, "zone": zone,
        "origin_day": str(origin), "stop_day_exclusive": str(stop),
        "training_first_day": str(first),
        "training_last_day": str(origin - timedelta(days=1)),
        "training_days": (origin - first).days,
        "training_hours": len(train_index), "prediction_hours": len(current_index),
        "training_labels_before_origin": True,
        "daily_features_keep_their_own_D_minus_1_08_origin": True,
        "training_features_sha256": frame_hash(train_x),
        "training_target_sha256": frame_hash(target),
        "prediction_features_sha256": frame_hash(current_x),
        "config": {"estimator": "hist", "target_mode": target_mode,
                   "iterations": 400, "depth": 6, "learning_rate": .035,
                   "l2_leaf_reg": 20., "seed": 20260925,
                   "clip_residual": 60.},
        "parameters": dict(PARAMETERS), "Storm_used_as_input": False,
    }
    return fitted, pd.DataFrame({"point": point}, index=current_index), audit


def predict_saved_block(fitted: HGBFit, current_features: pd.DataFrame,
                        current_nyx: pd.Series) -> pd.DataFrame:
    """Predict with an already fitted weekly HGB expert on an aligned interval."""
    if not isinstance(fitted, HGBFit) or fitted.variant not in VARIANTS:
        raise ValueError("A fitted archived HGB expert is required")
    if not isinstance(current_features, pd.DataFrame) or tuple(current_features.columns) != fitted.feature_columns:
        raise ValueError("Archived HGB feature order/schema changed")
    _check_index(current_features.index)
    if (not isinstance(current_nyx, pd.Series)
            or not current_nyx.index.equals(current_features.index)
            or not np.isfinite(current_nyx.to_numpy(dtype=float)).all()):
        raise ValueError("Aligned finite NYX point required")
    if (not len(current_features) or current_features.index[0] < civil_grid(
            fitted.origin_day, fitted.stop_day_exclusive, TIMEZONES[fitted.zone])[0]
            or not current_features.index.isin(civil_grid(
                fitted.origin_day, fitted.stop_day_exclusive, TIMEZONES[fitted.zone])).all()):
        raise ValueError("Prediction interval exceeds the frozen weekly fit")
    matrix = current_features.astype(float)
    if np.isinf(matrix.to_numpy()).any():
        raise ValueError("Infinite HGB feature")
    with threadpool_limits(limits=1):
        raw = fitted.model.predict(matrix)
    point = (current_nyx.to_numpy(dtype=float) + np.clip(raw, -60., 60.)
             if VARIANTS[fitted.variant][1] == "residual" else raw)
    if not np.isfinite(point).all():
        raise ValueError("Nonfinite HGB point")
    return pd.DataFrame({"point": point}, index=matrix.index)


def compose_warmup_reference(
    residual: pd.Series, absolute: pd.Series, augmented: pd.Series,
    nyx_point: pd.Series, test2_point: pd.Series,
) -> pd.DataFrame:
    """The archived ``three_hgb_mean__clip20__w0p75`` point, before gates."""
    series = (residual, absolute, augmented, nyx_point, test2_point)
    if any(not isinstance(item, pd.Series) for item in series):
        raise TypeError("Five aligned point series required")
    index = series[0].index
    _check_index(index)
    if any(not item.index.equals(index) or not np.isfinite(item.to_numpy(dtype=float)).all()
           for item in series):
        raise ValueError("Aligned finite point series required")
    three = (residual + absolute + augmented) / 3.
    clip20 = nyx_point + np.clip(test2_point - nyx_point, -20., 20.)
    mixed = .75 * three + .25 * clip20
    return pd.DataFrame({"three_hgb_mean": three,
                         "three_hgb_mean__clip20__w0p75": mixed}, index=index)
