"""Weekly CPU Test2 fit with daily D-1 predictions from supplied matrices.

This reproduces the numerical regime model used by the annual CWE reference,
while separating fit from prediction. Feature production, source vintages and
model checkpoint storage remain the caller's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import solar_wind_scarcity_regime as regime


PAIRS = (("BE", "FR"), ("DE", "NL"))
TIMEZONES = {"BE": "Europe/Brussels", "FR": "Europe/Paris",
             "DE": "Europe/Berlin", "NL": "Europe/Amsterdam"}
DAILY_PEAK_COLUMNS = (
    "own_daily_peak_residual_stress", "own_daily_peak_deficit_stress",
    "other_daily_peak_residual_stress", "other_daily_peak_deficit_stress",
)
FORBIDDEN_INPUT_NAMES = ("actual", "storm", "observed", "target")


@dataclass
class _Expert:
    model: Any | None
    fixed_knots: np.ndarray | None
    count: int
    minimum: float | None
    maximum: float | None
    regime_name: str

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            if self.fixed_knots is None:
                point = regime.SPIKE_THRESHOLD + 1. if self.regime_name == "spike" else 0.
                return np.full((len(features), len(regime.EXPERT_PROBABILITIES)), point)
            return np.repeat(self.fixed_knots[None, :], len(features), axis=0)
        raw = np.asarray(self.model.predict(features), dtype=float)
        internal = np.sort(raw, axis=1)
        if self.regime_name == "normal":
            internal = np.minimum(internal, regime.SPIKE_THRESHOLD)
        else:
            internal = np.maximum(internal, np.nextafter(regime.SPIKE_THRESHOLD, np.inf))
        lower = np.minimum(self.minimum, internal[:, 0])
        upper = np.maximum(self.maximum, internal[:, -1])
        return np.column_stack([lower, internal, upper])


@dataclass
class FittedTest2:
    pair: tuple[str, str]
    origin_day: str
    feature_names: tuple[str, ...]
    gate: Any | None
    calibrator: Any | None
    empirical_probability: float | None
    normal: _Expert
    spike: _Expert
    audit: dict

    def predict_day(self, features: pd.DataFrame, nyx_q50: pd.Series, *,
                    zone: str, forecast_issued_at_utc: pd.Timestamp) -> pd.DataFrame:
        """Score one complete civil day with the origin-frozen CPU estimators."""
        if zone not in self.pair:
            raise ValueError("Prediction country is outside the fitted Test2 pair")
        _check_matrix(features, expected=self.feature_names)
        if (not isinstance(nyx_q50, pd.Series) or not nyx_q50.index.equals(features.index)
                or not np.isfinite(nyx_q50.to_numpy(dtype=float)).all()
                or not np.array_equal(features.baseline_p50.to_numpy(dtype=float),
                                      nyx_q50.to_numpy(dtype=float))):
            raise ValueError("The exact same aligned NYX P50 must be supplied")
        day_values = np.asarray(features.index.tz_convert(TIMEZONES[zone]).date)
        if len(set(day_values)) != 1:
            raise ValueError("Only one complete local delivery day may be scored")
        day = day_values[0]
        origin = pd.Timestamp(self.origin_day).date()
        if not origin <= day < origin + timedelta(days=7):
            raise ValueError("Delivery day is outside the fitted weekly origin")
        expected = pd.date_range(str(day), str(day + timedelta(days=1)),
                                 tz=TIMEZONES[zone], freq="h", inclusive="left").tz_convert("UTC")
        if not features.index.equals(expected):
            raise ValueError("All physical hours of the delivery day are required")
        cutoff = pd.Timestamp(f"{day - timedelta(days=1)} 08:00",
                              tz=TIMEZONES[zone]).tz_convert("UTC")
        issued = pd.Timestamp(forecast_issued_at_utc)
        if issued.tzinfo is None or issued.tz_convert("UTC") > cutoff:
            raise ValueError("Test2 source bundle was unavailable by D-1 08h local")
        data = features.to_numpy(dtype=float)
        if self.gate is None:
            probability = np.full(len(data), self.empirical_probability, dtype=float)
        elif self.calibrator is None:
            probability = self.gate.predict_proba(data)[:, 1]
        else:
            logits = np.asarray(self.gate.predict(data, prediction_type="RawFormulaVal"),
                                dtype=float).reshape(-1, 1)
            probability = self.calibrator.predict_proba(logits)[:, 1]
        normal = self.normal.predict(data)
        spike = self.spike.predict(data)
        residual_quantiles = regime.mixture_quantiles(normal, spike, probability)
        baseline = nyx_q50.to_numpy(dtype=float)
        result = pd.DataFrame({
            "test2__q10": baseline + residual_quantiles[:, 0],
            "test2__q50": baseline + residual_quantiles[:, 1],
            "test2__q90": baseline + residual_quantiles[:, 2],
            "spike_probability": probability,
        }, index=features.index)
        if (not np.isfinite(result.to_numpy(dtype=float)).all()
                or (np.diff(result[["test2__q10", "test2__q50", "test2__q90"]]
                            .to_numpy(dtype=float), axis=1) < 0).any()):
            raise ValueError("Invalid Test2 daily quantiles or probability")
        return result


def _check_matrix(frame: pd.DataFrame, *, expected: tuple[str, ...] | None = None):
    if (not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique
            or "baseline_p50" not in frame):
        raise ValueError("Nonempty unique Test2 matrix with baseline_p50 required")
    if expected is not None and tuple(frame.columns) != expected:
        raise ValueError("Test2 prediction feature order differs from fit")
    if any(any(word in str(name).lower() for word in FORBIDDEN_INPUT_NAMES)
           for name in frame.columns):
        raise ValueError("Observed price, target and Storm cannot be Test2 features")
    if any(name in frame.columns for name in DAILY_PEAK_COLUMNS):
        raise ValueError("Historical Test2 removes the four daily peak columns")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Finite numeric Test2 features required; no imputation")


def _fit_expert(train: np.ndarray, residual: np.ndarray, name: str,
                parameters: dict) -> _Expert:
    count = len(residual)
    varying = count > 1 and np.any(np.ptp(train, axis=0) > 0)
    if (count >= regime.MINIMUM_EXPERT_ROWS[name] and varying
            and np.ptp(residual) > 0):
        from catboost import CatBoostRegressor
        alpha = regime.EXPERT_PROBABILITIES[1:-1]
        model = CatBoostRegressor(
            loss_function="MultiQuantile:alpha=" + ",".join(str(x) for x in alpha),
            **parameters)
        model.fit(train, residual)
        return _Expert(model, None, count, float(residual.min()),
                       float(residual.max()), name)
    if count:
        knots = np.quantile(residual, regime.EXPERT_PROBABILITIES)
        return _Expert(None, knots, count, float(residual.min()),
                       float(residual.max()), name)
    return _Expert(None, None, 0, None, None, name)


def fit_test2_origin(train_X: pd.DataFrame, train_residual: Sequence[float],
                     train_days: Sequence, *, pair: tuple[str, str],
                     origin_day: str, iterations: int = 120,
                     threads: int = 1, seed: int = 20260923) -> FittedTest2:
    """Fit one weekly pair using 90–365 strictly earlier civil days on CPU."""
    if tuple(pair) not in PAIRS:
        raise ValueError("Use an explicit historical BE/FR or DE/NL pair")
    if threads != 1 or not 1 <= iterations <= 120:
        raise ValueError("One CPU thread and 1–120 trees required")
    _check_matrix(train_X)
    origin = pd.Timestamp(origin_day).date()
    residual = np.asarray(train_residual, dtype=float)
    days = pd.DatetimeIndex(pd.to_datetime(list(train_days)))
    if (residual.shape != (len(train_X),) or len(days) != len(train_X)
            or days.hasnans or not np.isfinite(residual).all()):
        raise ValueError("Finite aligned training residuals and civil days required")
    civil = np.asarray(days.date)
    if (np.any(civil >= origin) or np.any(civil < origin - timedelta(days=365))
            or len(set(civil)) < 90):
        raise ValueError("Training days must be 90–365 strictly before the weekly origin")
    data = train_X.to_numpy(dtype=float)
    target = (residual > regime.SPIKE_THRESHOLD).astype(int)
    positives, negatives = int(target.sum()), int((1 - target).sum())
    parameters = {"iterations": int(iterations), "depth": 4, "learning_rate": .05,
                  "l2_leaf_reg": 3.0, "random_seed": int(seed),
                  "thread_count": 1, "allow_writing_files": False,
                  "verbose": False, "task_type": "CPU"}
    gate = calibrator = None
    empirical_probability = None
    calibration_mode = "empirical_frequency"
    varying = np.any(np.ptp(data, axis=0) > 0)
    if (positives < regime.MINIMUM_GATE_SPIKES
            or negatives < regime.MINIMUM_GATE_NORMAL or not varying):
        empirical_probability = positives / len(target)
    else:
        from catboost import CatBoostClassifier
        calibration = civil >= origin - timedelta(days=regime.CALIBRATION_DAYS)
        earlier = ~calibration
        enough = (int(target[earlier].sum()) >= regime.MINIMUM_GATE_SPIKES
                  and int((1 - target[earlier]).sum()) >= regime.MINIMUM_GATE_NORMAL
                  and int(target[calibration].sum()) >= regime.MINIMUM_GATE_SPIKES
                  and int((1 - target[calibration]).sum()) >= regime.MINIMUM_GATE_NORMAL
                  and np.any(np.ptp(data[earlier], axis=0) > 0))
        gate = CatBoostClassifier(loss_function="Logloss", **parameters)
        if enough:
            from sklearn.linear_model import LogisticRegression
            gate.fit(data[earlier], target[earlier])
            logits = np.asarray(gate.predict(data[calibration],
                                             prediction_type="RawFormulaVal"),
                                dtype=float).reshape(-1, 1)
            calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                            random_state=int(seed))
            calibrator.fit(logits, target[calibration])
            calibration_mode = "catboost_platt_calibrated"
        else:
            gate.fit(data, target)
            calibration_mode = "catboost_uncalibrated"
    normal = _fit_expert(data[target == 0], residual[target == 0], "normal", parameters)
    spike = _fit_expert(data[target == 1], residual[target == 1], "spike", parameters)
    return FittedTest2(
        pair=tuple(pair), origin_day=str(origin), feature_names=tuple(train_X.columns),
        gate=gate, calibrator=calibrator,
        empirical_probability=empirical_probability, normal=normal, spike=spike,
        audit={"origin_day": str(origin), "pair": list(pair),
               "train_first_day": str(min(civil)), "train_last_day": str(max(civil)),
               "train_unique_days": len(set(civil)), "train_rows": len(data),
               "feature_names": list(train_X.columns), "parameters": parameters,
               "gate_method": calibration_mode, "spike_rows": positives,
               "normal_rows": negatives, "future_labels_used": False,
               "forecast_source_vintages_certified": False})
