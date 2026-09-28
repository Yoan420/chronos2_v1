"""Standalone CPU retraining and chronological evaluation for P(price < 0).

All feature matrices and observed prices are supplied by the caller. This module
does not read the archived research runs, fetch Saturn data, or publish a live
forecast. Feature publication timing remains the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL = "nyx_negative_price_fixed_platt_cpu_v1"
ZONES = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
}
FEATURE_COUNT = 123
ALERT_THRESHOLD = 0.5
SEED = 20260927


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _day(value: str | date) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    _require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None, "A YYYY-MM-DD civil day is required")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("A YYYY-MM-DD civil day is required") from exc


def physical_grid(first_day: str | date, stop_day: str | date, zone: str) -> pd.DatetimeIndex:
    """Return each real UTC hour in a local civil-day interval, including DST."""

    _require(zone in ZONES, "Unsupported CWE country")
    first, stop = _day(first_day), _day(stop_day)
    _require(first < stop, "A nonempty civil-day interval is required")
    return pd.date_range(str(first), str(stop), tz=ZONES[zone], freq="h", inclusive="left").tz_convert("UTC")


def _validated_features(frame: pd.DataFrame, expected: pd.DatetimeIndex, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    _require(
        isinstance(frame, pd.DataFrame)
        and isinstance(frame.index, pd.DatetimeIndex)
        and str(frame.index.tz) == "UTC"
        and frame.index.equals(expected)
        and frame.index.is_unique,
        "Features require the exact physical UTC grid",
    )
    _require(
        len(frame.columns) == FEATURE_COUNT
        and frame.columns.is_unique
        and all(isinstance(name, str) and name for name in frame.columns)
        and not any(
            any(forbidden in name.lower() for forbidden in ("storm", "actual", "observed", "target"))
            for name in frame.columns
        ),
        "Exactly 123 distinct non-label feature columns are required",
    )
    _require(columns is None or tuple(frame.columns) == columns, "Feature column order changed")
    _require(
        all(pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) for dtype in frame.dtypes),
        "Feature columns must be numeric or boolean",
    )
    values = frame.astype(np.float64)
    _require(not np.isinf(values.to_numpy()).any(), "Infinite feature values are forbidden")
    flags = [name for name in values if name.endswith("__available")]
    _require(not flags or values[flags].isin((0.0, 1.0)).all().all(), "Availability flags must be binary")
    return values


def _validated_actual(actual: pd.Series, expected: pd.DatetimeIndex) -> pd.Series:
    _require(
        isinstance(actual, pd.Series)
        and isinstance(actual.index, pd.DatetimeIndex)
        and str(actual.index.tz) == "UTC"
        and actual.index.equals(expected)
        and "storm" not in str(actual.name).lower()
        and pd.api.types.is_numeric_dtype(actual.dtype)
        and not pd.api.types.is_bool_dtype(actual.dtype),
        "Observed historical prices require the exact past UTC grid",
    )
    values = actual.astype(np.float64)
    _require(np.isfinite(values.to_numpy()).all(), "Observed historical prices must be finite")
    return values


def catboost_parameters(threads: int = 2) -> dict[str, Any]:
    _require(type(threads) is int and threads in (1, 2), "One or two CPU threads are required")
    return {
        "loss_function": "Logloss",
        "iterations": 120,
        "depth": 4,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_seed": SEED,
        "task_type": "CPU",
        "thread_count": threads,
        "nan_mode": "Min",
        "verbose": False,
        "allow_writing_files": False,
        "use_best_model": False,
    }


@dataclass
class BlockResult:
    estimator: Any | None
    probabilities: pd.DataFrame
    audit: dict[str, Any]


def fit_predict_block(
    history_features: pd.DataFrame,
    observed_history: pd.Series,
    future_features: pd.DataFrame,
    *,
    zone: str,
    origin_day: str | date,
    stop_day: str | date,
    threads: int = 2,
) -> BlockResult:
    """Fit on 337 past days, calibrate on 28 past days, predict up to 7 days."""

    from catboost import CatBoostClassifier
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    params = catboost_parameters(threads)
    origin, stop = _day(origin_day), _day(stop_day)
    _require(origin < stop <= origin + timedelta(days=7), "Prediction block must span one to seven civil days")
    first, calibration_first = origin - timedelta(days=365), origin - timedelta(days=28)
    history_grid = physical_grid(first, origin, zone)
    prediction_grid = physical_grid(origin, stop, zone)
    hx = _validated_features(history_features, history_grid)
    cx = _validated_features(future_features, prediction_grid, tuple(hx.columns))
    actual = _validated_actual(observed_history, history_grid)
    labels = (actual < 0.0).astype(np.int8)
    fit_grid = physical_grid(first, calibration_first, zone)
    cal_grid = physical_grid(calibration_first, origin, zone)
    fit_x, fit_y = hx.loc[fit_grid], labels.loc[fit_grid]
    cal_x, cal_y = hx.loc[cal_grid], labels.loc[cal_grid]

    started = perf_counter()
    estimator: Any | None = None
    fallback: str | None = None
    calibration = "frequency_fallback"
    slope: float | None = None
    intercept: float | None = None
    if fit_y.nunique() < 2:
        fallback = "one_class"
    elif not bool((fit_x.nunique(dropna=False) > 1).any()):
        fallback = "constant_features"
    if fallback is None:
        estimator = CatBoostClassifier(**params)
        with threadpool_limits(limits=threads):
            estimator.fit(fit_x, fit_y)
        _require(
            int(estimator.tree_count_) == 120
            and list(estimator.classes_) == [0, 1]
            and list(estimator.feature_names_) == list(fit_x.columns),
            "Fitted CatBoost recipe or feature schema differs",
        )
        raw_matrix = np.asarray(estimator.predict_proba(cx, thread_count=threads), dtype=np.float64)
        _require(raw_matrix.shape == (len(cx), 2), "Binary CatBoost probability matrix required")
        raw = raw_matrix[:, 1].copy()
        if int(cal_y.sum()) >= 10 and int((1 - cal_y).sum()) >= 10:
            margins = np.asarray(estimator.predict(cal_x, prediction_type="RawFormulaVal", thread_count=threads), dtype=np.float64)
            _require(margins.shape == (len(cal_x),) and np.isfinite(margins).all(), "Finite calibration margins required")
            calibrator = LogisticRegression(
                C=1.0, fit_intercept=True, solver="lbfgs", max_iter=1000, random_state=SEED
            )
            with threadpool_limits(limits=threads):
                calibrator.fit(margins.reshape(-1, 1), cal_y)
            _require(list(calibrator.classes_) == [0, 1] and int(calibrator.n_iter_[0]) < 1000, "Platt fit did not converge")
            slope, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
            _require(np.isfinite([slope, intercept]).all(), "Finite Platt coefficients required")
            prediction_margins = np.asarray(estimator.predict(cx, prediction_type="RawFormulaVal", thread_count=threads), dtype=np.float64)
            _require(prediction_margins.shape == (len(cx),) and np.isfinite(prediction_margins).all(), "Finite prediction margins required")
            final = expit(slope * prediction_margins + intercept)
            calibration = "platt"
        else:
            final = raw.copy()
            calibration = "raw_insufficient_calibration_classes"
    else:
        raw = np.full(len(cx), float(labels.mean()), dtype=np.float64)
        final = raw.copy()

    _require(
        raw.shape == final.shape == (len(cx),)
        and np.isfinite(raw).all()
        and np.isfinite(final).all()
        and ((raw >= 0.0) & (raw <= 1.0)).all()
        and ((final >= 0.0) & (final <= 1.0)).all(),
        "Predicted probabilities must be finite and in [0,1]",
    )
    probabilities = pd.DataFrame(
        {
            "p_negative_raw": raw,
            "p_negative": final,
            "is_negative_predicted": final >= ALERT_THRESHOLD,
            "timestamp_local": [timestamp.isoformat() for timestamp in cx.index.tz_convert(ZONES[zone])],
        },
        index=cx.index.copy(),
    ).rename_axis("timestamp_utc")
    audit = {
        "protocol": PROTOCOL,
        "event": "observed_price < 0.0",
        "alert_threshold": ALERT_THRESHOLD,
        "zone": zone,
        "origin_day": str(origin),
        "stop_day_exclusive": str(stop),
        "history_first_day": str(first),
        "calibration_first_day": str(calibration_first),
        "history_civil_days": 365,
        "classifier_civil_days": 337,
        "calibration_civil_days": 28,
        "history_hours": len(hx),
        "fit_hours": len(fit_x),
        "calibration_hours": len(cal_x),
        "prediction_hours": len(cx),
        "fit_negative_hours": int(fit_y.sum()),
        "calibration_negative_hours": int(cal_y.sum()),
        "history_negative_hours": int(labels.sum()),
        "history_event_frequency": float(labels.mean()),
        "feature_columns": list(hx.columns),
        "parameters": params,
        "platt_parameters": {"C": 1.0, "fit_intercept": True, "solver": "lbfgs", "max_iter": 1000, "random_state": SEED},
        "calibration_minimum_events_per_class": 10,
        "fallback_reason": fallback,
        "calibration_status": calibration,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "models_fitted": int(estimator is not None),
        "tree_count": int(estimator.tree_count_) if estimator is not None else 0,
        "fit_seconds": perf_counter() - started,
        "all_training_labels_before_origin": True,
        "forecast_labels_used": False,
        "storm_used": False,
    }
    return BlockResult(estimator, probabilities, audit)


def probability_metrics(observed_price: pd.Series, p_negative: pd.Series) -> dict[str, Any]:
    """Score predictions at their physical UTC hours, with a fixed 0.5 alert."""

    from sklearn.metrics import average_precision_score

    _require(observed_price.index.equals(p_negative.index), "Observed and predicted grids must match")
    actual = _validated_actual(observed_price, p_negative.index)
    p = p_negative.to_numpy(dtype=np.float64)
    _require(np.isfinite(p).all() and ((p >= 0.0) & (p <= 1.0)).all(), "Valid probabilities required")
    y = actual.to_numpy() < 0.0
    alert = p >= ALERT_THRESHOLD
    tp, fp = int((y & alert).sum()), int((~y & alert).sum())
    fn, tn = int((y & ~alert).sum()), int((~y & ~alert).sum())
    clipped = np.clip(p, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
    return {
        "hours": len(p),
        "negative_hours": int(y.sum()),
        "brier": float(np.mean((p - y.astype(float)) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (~y) * np.log1p(-clipped))),
        "average_precision": float(average_precision_score(y, p)) if y.any() else None,
        "threshold": ALERT_THRESHOLD,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


@dataclass
class ChronologicalEvaluation:
    probabilities: pd.DataFrame
    metrics: dict[str, Any]
    block_audits: list[dict[str, Any]]


def evaluate_weekly(
    features: pd.DataFrame,
    observed_price: pd.Series,
    *,
    zone: str,
    first_origin_day: str | date,
    stop_day: str | date,
    threads: int = 2,
) -> ChronologicalEvaluation:
    """Walk weekly origins, fitting only on the 365 days preceding each origin."""

    first_origin, stop = _day(first_origin_day), _day(stop_day)
    _require(first_origin < stop, "Evaluation period must be nonempty")
    first_source = first_origin - timedelta(days=365)
    complete_grid = physical_grid(first_source, stop, zone)
    complete_features = _validated_features(features, complete_grid)
    complete_actual = _validated_actual(observed_price, complete_grid)
    blocks: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    origin = first_origin
    while origin < stop:
        block_stop = min(origin + timedelta(days=7), stop)
        hx = complete_features.loc[physical_grid(origin - timedelta(days=365), origin, zone)]
        hy = complete_actual.loc[hx.index]
        cx = complete_features.loc[physical_grid(origin, block_stop, zone)]
        block = fit_predict_block(hx, hy, cx, zone=zone, origin_day=origin, stop_day=block_stop, threads=threads)
        blocks.append(block.probabilities.assign(origin_day=str(origin)))
        audits.append(block.audit)
        origin = block_stop
    predictions = pd.concat(blocks)
    scored_grid = physical_grid(first_origin, stop, zone)
    _require(predictions.index.equals(scored_grid), "Chronological predictions must cover each evaluated hour once")
    metrics = probability_metrics(complete_actual.loc[scored_grid], predictions["p_negative"])
    metrics.update(
        zone=zone,
        first_origin_day=str(first_origin),
        stop_day_exclusive=str(stop),
        origins=len(audits),
        models_fitted=sum(audit["models_fitted"] for audit in audits),
        retrospective_evaluation=True,
        independent_validation=False,
    )
    return ChronologicalEvaluation(predictions, metrics, audits)
