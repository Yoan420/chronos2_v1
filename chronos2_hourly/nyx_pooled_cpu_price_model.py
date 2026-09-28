"""CPU port of the pooled CWE price experts for prospective retraining.

The feature, target, base and physical-hour contracts match the historical GPU
producer. CPU fits have a distinct protocol and must be evaluated separately:
changing CatBoost's task type does not reproduce the audited GPU predictions.
Saved GPU CBMs can nevertheless be loaded for CPU inference when their ordered
feature schema and tree count match the requested block.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import tempfile
import time

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

PROTOCOL = "nyx_pooled_cwe_cpu_rmse_v1"
ZONES = ("FR", "DE", "BE", "NL")


def frame_hash(frame):
    """Use the exact historical digest of values, index and ordered names."""
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    names = list(frame.columns) if isinstance(frame, pd.DataFrame) else [frame.name]
    return hashlib.sha256(values + json.dumps(names).encode()).hexdigest()


def grid(first, stop, timezone="Europe/Paris"):
    """Return all physical UTC hours in a pair of local civil dates."""
    return pd.date_range(pd.Timestamp(first, tz=timezone), pd.Timestamp(stop, tz=timezone),
                         freq="h", inclusive="left").tz_convert("UTC")


@dataclass(frozen=True)
class PooledConfig:
    target_mode: str = "residual"
    iterations: int = 1000
    depth: int = 7
    learning_rate: float = .035
    l2_leaf_reg: float = 10.
    seed: int = 20260925


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parameters(config: PooledConfig, thread_count: int = 2) -> dict:
    """Keep the fixed numerical recipe, replacing GPU-only settings with CPU."""
    require(isinstance(config, PooledConfig) and config.target_mode in ("residual", "absolute"),
            "Explicit pooled target required")
    require(type(config.iterations) is int and 1 <= config.iterations <= 2000
            and type(config.depth) is int and 1 <= config.depth <= 10
            and np.isfinite(config.learning_rate) and 0 < config.learning_rate <= 1
            and np.isfinite(config.l2_leaf_reg) and config.l2_leaf_reg >= 0,
            "Invalid pooled recipe")
    require(type(thread_count) is int and 1 <= thread_count <= 128, "Positive CPU thread count required")
    return {"loss_function": "RMSE", "eval_metric": "RMSE", "iterations": config.iterations,
            "depth": config.depth, "learning_rate": config.learning_rate,
            "l2_leaf_reg": config.l2_leaf_reg, "random_seed": config.seed,
            "task_type": "CPU", "thread_count": thread_count, "verbose": False,
            "allow_writing_files": False, "nan_mode": "Min", "boosting_type": "Plain",
            "grow_policy": "SymmetricTree", "bootstrap_type": "Bayesian",
            "bagging_temperature": 0., "border_count": 254,
            "leaf_estimation_method": "Newton", "leaf_estimation_iterations": 1,
            "score_function": "L2", "random_strength": 1., "boost_from_average": True,
            "has_time": True, "use_best_model": False}


def _validate_pooled_index(value, times, name):
    expected = pd.MultiIndex.from_product([times, ZONES], names=["timestamp_utc", "zone"])
    require(isinstance(value.index, pd.MultiIndex) and value.index.nlevels == 2
            and value.index.names == expected.names and value.index.is_unique
            and value.index.equals(expected), f"{name}: invalid physical UTC/country MultiIndex")


def _versions():
    return {"python": platform.python_version(),
            **{name: version(name) for name in ("catboost", "numpy", "pandas")}}


def build_inputs(features, actual, nyx_point, *, origin_day, stop_day,
                 config=PooledConfig(), initial_training_day="2024-09-23",
                 thread_count=2):
    """Build exactly the historical ordered matrices; only audit/fit identity differs."""
    params = parameters(config, thread_count)
    require(all(isinstance(v, Mapping) for v in (features, actual, nyx_point))
            and set(features) == set(actual) == set(nyx_point) == set(ZONES),
            "Exactly four CWE countries required")
    origin, stop, initial = (pd.Timestamp(v).date() for v in (origin_day, stop_day, initial_training_day))
    require(origin < stop <= origin + timedelta(days=7), "One to seven prediction days required")
    first = max(initial, origin - timedelta(days=365))
    days = (origin - first).days
    require(days >= 90 and (origin < pd.Timestamp("2025-09-24").date() or days == 365),
            "Invalid chronological training window")
    train_index, current_index = grid(first, origin), grid(origin, stop)
    for frame in features.values():
        require(isinstance(frame, pd.DataFrame) and not frame.empty and frame.columns.is_unique
                and all(isinstance(column, str) for column in frame.columns),
                "Unique named feature table required")
    common_index = features[ZONES[0]].index
    columns = sorted({name for frame in features.values() for name in frame.columns})
    indicators = ["country_" + zone for zone in ZONES]
    require(not set(indicators) & set(columns), "Country indicator collision")
    train_frames, current_frames, labels, bases = {}, {}, {}, {}
    structural_absence = {}
    for zone in ZONES:
        frame, label, base = features[zone], actual[zone], nyx_point[zone]
        require(not any(any(term in column.lower() for term in
                            ("storm", "actual", "observed", "target")) for column in frame.columns),
                "Labels and Storm are forbidden feature columns")
        index = frame.index
        require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC" and not index.hasnans
                and index.is_unique and index.is_monotonic_increasing and index.equals(index.floor("h"))
                and index.equals(common_index), "Identical ordered physical UTC grids required")
        for value in (label, base):
            require(isinstance(value, pd.Series) and value.index.equals(index)
                    and "storm" not in str(value.name).lower(),
                    "Aligned observation/NYX series required")
        require(train_index.isin(index).all() and current_index.isin(index).all(),
                "Incomplete civil grid")
        structural_absence[zone] = sorted(set(columns) - set(frame.columns))
        for selected, destination in ((train_index, train_frames), (current_index, current_frames)):
            block = frame.loc[selected].reindex(columns=columns).astype(float)
            for column in structural_absence[zone]:
                if column.endswith("__available"):
                    block[column] = 0.
            require(not np.isinf(block.to_numpy()).any(), "Infinite feature value")
            flags = [column for column in columns if column.endswith("__available")]
            require(not flags or block[flags].isin([0., 1.]).all().all(),
                    "Availability indicators must be finite binary values")
            for country in ZONES:
                block["country_" + country] = float(country == zone)
            destination[zone] = block
        past_y = label.loc[train_index].astype(float)
        past_base = base.loc[train_index].astype(float)
        current_base = base.loc[current_index].astype(float)
        require(all(np.isfinite(value.to_numpy()).all() for value in
                    (past_y, past_base, current_base)), "Finite past labels and NYX required")
        with np.errstate(over="ignore", invalid="ignore"):
            labels[zone] = past_y - past_base if config.target_mode == "residual" else past_y
        require(np.isfinite(labels[zone].to_numpy()).all(), "Nonfinite residual target")
        bases[zone] = current_base

    def pool(parts, times):
        index = pd.MultiIndex.from_product([times, ZONES], names=["timestamp_utc", "zone"])
        return pd.concat(parts, names=["zone", "timestamp_utc"]).swaplevel().reindex(index)

    train_x, current_x = pool(train_frames, train_index), pool(current_frames, current_index)
    target = pool(labels, train_index).rename("target")
    current_base = pool(bases, current_index).rename("nyx_point")
    for value, times, name in ((train_x, train_index, "training features"),
                               (target, train_index, "training target"),
                               (current_x, current_index, "prediction features"),
                               (current_base, current_index, "prediction NYX")):
        _validate_pooled_index(value, times, name)
    require(list(train_x.columns) == list(current_x.columns),
            "Training/prediction feature schema differs")
    audit = {"protocol": PROTOCOL, "config": asdict(config), "parameters": params,
             "origin_day": str(origin), "stop_day_exclusive": str(stop),
             "training_first_day": str(first),
             "training_last_day": str(origin - timedelta(days=1)), "training_days": days,
             "training_hours_per_country": len(train_index), "training_rows": len(train_x),
             "prediction_hours_per_country": len(current_index), "prediction_rows": len(current_x),
             "feature_columns": list(train_x.columns),
             "structurally_absent_features": structural_absence,
             "row_order": "physical UTC hour, then FR DE BE NL",
             "pooling": "equal weight per country-hour",
             "structural_missing_value": "NaN", "structural_availability_value": 0.,
             "training_features_sha256": frame_hash(train_x),
             "training_target_sha256": frame_hash(target),
             "prediction_features_sha256": frame_hash(current_x),
             "prediction_base_sha256": frame_hash(current_base),
             "training_labels_before_origin": True,
             "daily_features_keep_their_own_D_minus_1_08_origin": True,
             "Storm_used_as_input": False, "target_clipped": False,
             "prediction_clipped": False, "GPU_historical_scores_reproduced": False,
             "dependency_versions": _versions(),
             "model_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return train_x, target, current_x, current_base, audit


def _save_model(estimator: CatBoostRegressor, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, filename = tempfile.mkstemp(prefix=destination.stem + ".", suffix=".tmp.cbm",
                                        dir=destination.parent)
    os.close(handle)
    temporary = Path(filename)
    try:
        estimator.save_model(str(temporary), format="cbm")
        data = temporary.read_bytes()
        require(bool(data), "Empty saved CBM")
        require(not destination.exists(), "Model destination already exists")
        os.replace(temporary, destination)
        require(destination.read_bytes() == data, "Published CBM differs from saved bytes")
        return {"saved": True, "path": str(destination), "format": "cbm",
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    finally:
        temporary.unlink(missing_ok=True)


def _point_frames(estimator: CatBoostRegressor, current_x: pd.DataFrame,
                  base: pd.Series, config: PooledConfig, thread_count: int) -> tuple[dict, pd.DataFrame]:
    point = np.asarray(estimator.predict(current_x, task_type="CPU", thread_count=thread_count),
                       dtype=float)
    require(point.ndim == 1 and len(point) == len(current_x),
            "Unexpected pooled prediction shape")
    with np.errstate(over="ignore", invalid="ignore"):
        if config.target_mode == "residual":
            point = base.to_numpy() + point
    require(np.isfinite(point).all(), "Nonfinite pooled prediction")
    frame = pd.DataFrame({"point": point}, index=current_x.index)
    return {zone: frame.xs(zone, level="zone").copy() for zone in ZONES}, frame


def fit_pooled_block(features, actual, nyx_point, *, origin_day, stop_day,
                     config=PooledConfig(), model_path=None,
                     initial_training_day="2024-09-23", thread_count=2):
    """Fit a new CPU expert; returns the historical per-country point shape."""
    started = time.monotonic()
    train_x, target, current_x, base, audit = build_inputs(
        features, actual, nyx_point, origin_day=origin_day, stop_day=stop_day,
        config=config, initial_training_day=initial_training_day,
        thread_count=thread_count)
    destination = None if model_path is None else Path(model_path).resolve()
    if destination is not None:
        require(destination.suffix.lower() == ".cbm", "A .cbm model path is required")
        require(not destination.exists(), "Model destination already exists")
    estimator = CatBoostRegressor(**parameters(config, thread_count))
    estimator.fit(train_x, target)
    output, frame = _point_frames(estimator, current_x, base, config, thread_count)
    audit["fitted_parameters"] = estimator.get_all_params()
    audit["tree_count"] = int(estimator.tree_count_)
    audit["prediction_points_sha256"] = frame_hash(frame)
    audit["model"] = ({"saved": False, "path": None, "format": "cbm", "sha256": None, "bytes": None}
                      if destination is None else _save_model(estimator, destination))
    audit["model_sha256"] = audit["model"]["sha256"]
    audit["fit_seconds"] = time.monotonic() - started
    return output, audit


def predict_saved_block(model_path, features, actual, nyx_point, *, origin_day, stop_day,
                        config=PooledConfig(), initial_training_day="2024-09-23",
                        thread_count=2, expected_model_sha256=None):
    """Infer on CPU from a CPU or historical GPU CBM, with exact input checks.

    This performs no fit. The caller must provide the complete, chronologically
    eligible 365-day histories and the current features/base for all four zones.
    """
    _, _, current_x, base, audit = build_inputs(
        features, actual, nyx_point, origin_day=origin_day, stop_day=stop_day,
        config=config, initial_training_day=initial_training_day,
        thread_count=thread_count)
    path = Path(model_path).resolve()
    require(path.is_file() and path.suffix.lower() == ".cbm", "Existing CBM required")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(expected_model_sha256 is None or digest == expected_model_sha256,
            "CBM SHA-256 differs")
    estimator = CatBoostRegressor()
    estimator.load_model(str(path), format="cbm")
    require(int(estimator.tree_count_) == config.iterations, "CBM tree count differs")
    require(list(estimator.feature_names_) == list(current_x.columns),
            "CBM ordered feature schema differs")
    output, frame = _point_frames(estimator, current_x, base, config, thread_count)
    audit["prediction_points_sha256"] = frame_hash(frame)
    audit["tree_count"] = int(estimator.tree_count_)
    audit["model_sha256"] = digest
    audit["inference_task_type"] = "CPU"
    audit["models_fitted"] = 0
    return output, audit
