"""Fixed country routing for the best audited annual RMSE candidates.

This only combines already produced expert forecasts. It does not train models,
obtain covariates, or claim that a CPU retrain has the historical GPU scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


VARIANTS = {
    "FR": "residual__disagreement20__w1p0",
    "DE": "boosting_2000_mean_disagreement20",
    "BE": "boosting_2000_mean_disagreement20",
    "NL": "boosting_2000_mean_all",
}


def _check(points: pd.DataFrame, columns: tuple[str, ...]) -> None:
    if not isinstance(points, pd.DataFrame) or tuple(points.columns) != columns:
        raise ValueError(f"Expected exactly these columns in order: {columns}")
    index = points.index
    if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC"
            or index.empty or index.hasnans or not index.is_unique
            or not index.is_monotonic_increasing or not index.equals(index.floor("h"))):
        raise ValueError("Expected unique, ordered physical UTC hours")
    if not np.isfinite(points.to_numpy(dtype=float)).all():
        raise ValueError("Forecast points must be finite")


def select_country(zone: str, points: pd.DataFrame) -> pd.Series:
    """Return the fixed historical choice without labels or Storm inputs.

    `reference` must be produced by the independently selected reference
    pipeline; this function does not substitute a different live baseline.
    """
    if zone == "FR":
        _check(points, ("reference", "residual"))
        reference = points["reference"].to_numpy(dtype=float)
        residual = points["residual"].to_numpy(dtype=float)
        selected = np.where(np.abs(residual - reference) >= 20., residual, reference)
    elif zone in ("DE", "BE", "NL"):
        _check(points, ("reference", "residual_2000", "absolute_2000"))
        reference = points["reference"].to_numpy(dtype=float)
        mean = (points["residual_2000"].to_numpy(dtype=float)
                + points["absolute_2000"].to_numpy(dtype=float)) / 2.
        selected = (np.where(np.abs(mean - reference) >= 20., mean, reference)
                    if zone in ("DE", "BE") else mean)
    else:
        raise ValueError("Only the audited FR, DE, BE and NL selections are supported")
    return pd.Series(selected, index=points.index, name="nyx_regional_rmse_point")
