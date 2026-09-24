from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .common import DEFAULT_QUANTILES, safe_ratio


def correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return math.nan
    return float(np.corrcoef(actual, predicted)[0, 1])


def pinball_loss(
    actual: np.ndarray,
    forecast: np.ndarray,
    quantile: float,
) -> float:
    error = actual - forecast
    return float(
        np.mean(
            np.maximum(
                quantile * error,
                (quantile - 1.0) * error,
            )
        )
    )


def compute_metrics(
    predictions: pd.DataFrame,
    extreme_threshold: float,
) -> dict[str, Any]:
    actual = predictions["actual"].to_numpy(dtype=np.float64)
    q10 = predictions["q10"].to_numpy(dtype=np.float64)
    q50 = predictions["q50"].to_numpy(dtype=np.float64)
    q90 = predictions["q90"].to_numpy(dtype=np.float64)
    point = predictions["point"].to_numpy(dtype=np.float64)
    error = q50 - actual

    negative_actual = actual < 0
    negative_pred = q50 < 0
    neg_tp = int(np.sum(negative_actual & negative_pred))
    neg_fp = int(np.sum(~negative_actual & negative_pred))
    neg_fn = int(np.sum(negative_actual & ~negative_pred))

    extreme_actual = np.abs(actual) >= extreme_threshold
    extreme_pred = np.abs(q50) >= extreme_threshold
    ext_tp = int(np.sum(extreme_actual & extreme_pred))
    ext_fp = int(np.sum(~extreme_actual & extreme_pred))
    ext_fn = int(np.sum(extreme_actual & ~extreme_pred))

    pinballs = []
    metrics: dict[str, Any] = {
        "n": int(len(actual)),
        "mae_q50": float(np.mean(np.abs(error))),
        "rmse_q50": float(np.sqrt(np.mean(error**2))),
        "bias_q50": float(np.mean(error)),
        "median_ae_q50": float(np.median(np.abs(error))),
        "correlation_q50": correlation(actual, q50),
        "mae_point": float(np.mean(np.abs(point - actual))),
        "coverage_q10_q90": float(
            np.mean((actual >= q10) & (actual <= q90))
        ),
        "interval_width_q10_q90": float(np.mean(q90 - q10)),
        "negative_precision": safe_ratio(neg_tp, neg_tp + neg_fp),
        "negative_recall": safe_ratio(neg_tp, neg_tp + neg_fn),
        "extreme_precision": safe_ratio(ext_tp, ext_tp + ext_fp),
        "extreme_recall": safe_ratio(ext_tp, ext_tp + ext_fn),
        "actual_negative": int(np.sum(negative_actual)),
        "actual_extreme": int(np.sum(extreme_actual)),
    }
    for level in DEFAULT_QUANTILES:
        key = f"q{int(level * 100):02d}"
        value = pinball_loss(
            actual,
            predictions[key].to_numpy(dtype=np.float64),
            level,
        )
        metrics[f"pinball_{key}"] = value
        pinballs.append(value)
    metrics["mean_pinball_9q"] = float(np.mean(pinballs))
    metrics["crps_quantile_approx"] = float(2 * np.mean(pinballs))

    ramp_errors = []
    for _, block in predictions.groupby(
        "origin_timestamp",
        sort=False,
    ):
        block = block.sort_values("horizon_step")
        if len(block) >= 2:
            ramp_errors.append(
                np.diff(block["q50"].to_numpy(dtype=np.float64))
                - np.diff(block["actual"].to_numpy(dtype=np.float64))
            )
    if ramp_errors:
        ramps = np.concatenate(ramp_errors)
        metrics["ramp_mae"] = float(np.mean(np.abs(ramps)))
        metrics["ramp_rmse"] = float(np.sqrt(np.mean(ramps**2)))
        metrics["ramp_bias"] = float(np.mean(ramps))
    else:
        metrics["ramp_mae"] = math.nan
        metrics["ramp_rmse"] = math.nan
        metrics["ramp_bias"] = math.nan
    return metrics


def metric_breakdowns(
    native: pd.DataFrame,
    baseline: pd.DataFrame | None,
    extreme_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon_rows = []
    hour_rows = []

    for step, block in native.groupby("horizon_step", sort=True):
        row = {"horizon_step": int(step)}
        row.update(
            {
                f"native_{key}": value
                for key, value in compute_metrics(
                    block,
                    extreme_threshold,
                ).items()
            }
        )
        if baseline is not None:
            base = baseline.loc[baseline["horizon_step"].eq(step)]
            row.update(
                {
                    f"baseline_{key}": value
                    for key, value in compute_metrics(
                        base,
                        extreme_threshold,
                    ).items()
                }
            )
        horizon_rows.append(row)

    native_with_hour = native.copy()
    native_with_hour["delivery_hour"] = (
        native_with_hour["timestamp"].dt.hour
    )
    baseline_with_hour = baseline.copy() if baseline is not None else None
    if baseline_with_hour is not None:
        baseline_with_hour["delivery_hour"] = (
            baseline_with_hour["timestamp"].dt.hour
        )

    for hour, block in native_with_hour.groupby(
        "delivery_hour",
        sort=True,
    ):
        row = {"delivery_hour": int(hour)}
        row.update(
            {
                f"native_{key}": value
                for key, value in compute_metrics(
                    block,
                    extreme_threshold,
                ).items()
            }
        )
        if baseline_with_hour is not None:
            base = baseline_with_hour.loc[
                baseline_with_hour["delivery_hour"].eq(hour)
            ]
            row.update(
                {
                    f"baseline_{key}": value
                    for key, value in compute_metrics(
                        base,
                        extreme_threshold,
                    ).items()
                }
            )
        hour_rows.append(row)

    return pd.DataFrame(horizon_rows), pd.DataFrame(hour_rows)


def gain_percent(
    baseline: float | None,
    native: float | None,
) -> float:
    if (
        baseline is None
        or native is None
        or not np.isfinite(baseline)
        or baseline == 0
    ):
        return math.nan
    return float(100 * (baseline - native) / baseline)


def add_comparison_fields(
    native: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    result = dict(native)
    if baseline is None:
        return result
    for key, value in baseline.items():
        result[f"baseline_{key}"] = value
    for metric in (
        "mae_q50",
        "rmse_q50",
        "mean_pinball_9q",
        "crps_quantile_approx",
        "ramp_mae",
    ):
        result[f"{metric}_gain_percent"] = gain_percent(
            baseline.get(metric),
            native.get(metric),
        )
    return result
