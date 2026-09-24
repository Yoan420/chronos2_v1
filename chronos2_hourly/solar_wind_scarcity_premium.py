"""Causal amplitude calibration for weak wind/solar forecasts, isolated from NYX.

This positive residual premium is fitted AFTER the sealed final baseline. It
does not retrain Chronos, CatBoost or Kalman. A nonnegative ridge least-squares
fit learns the missing price amplitude; zero coefficients remain possible.
See docs/solar_wind_scarcity_premium.md for the fixed experimental protocol.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.optimize import nnls


PROTOCOL_VERSION = "solar_wind_scarcity_premium_v1"
BASIS_COLUMNS = ("scarcity_base", "scarcity_demand", "scarcity_extreme")
FEATURE_COLUMNS = ("joint_deficit", "residual_stress", *BASIS_COLUMNS)
QUANTILES = ("q10", "q50", "q90")
BASELINE_COLUMNS = tuple("residual_kalman__" + q for q in QUANTILES)


def _daily_grid(frame: pd.DataFrame, timezone: str, name: str):
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError(f"{name}: nonempty DataFrame with unique columns required")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC":
        raise ValueError(f"{name}: explicit UTC DatetimeIndex required")
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(f"{name}: ordered, unique, finite UTC timestamps required")
    local_days = index.tz_convert(timezone).date
    days = list(dict.fromkeys(local_days))
    expected = pd.date_range(pd.Timestamp(days[0], tz=timezone),
                             pd.Timestamp(days[-1] + timedelta(days=1), tz=timezone),
                             freq="h", inclusive="left").tz_convert("UTC")
    if not index.equals(expected):
        raise ValueError(f"{name}: complete consecutive local delivery days required, including DST")
    boundaries = np.r_[0, np.flatnonzero(local_days[1:] != local_days[:-1]) + 1, len(index)]
    return days, boundaries


def _finite_columns(frame, columns, name):
    if not set(columns) <= set(frame.columns):
        raise ValueError(f"{name}: missing required columns {sorted(set(columns) - set(frame.columns))}")
    try:
        values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: numeric values required") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: finite values required; missing data are not imputed")
    return values


def build_scarcity_basis(covariates: pd.DataFrame, *, zone="DE", timezone="Europe/Berlin",
                         history_days=365, min_history_days=14):
    """Three monotone scarcity bases normalized on strictly earlier civil days.

    Only generation FORECASTS and forecast residual load in GW are used.
    Historical normalizations are fitted separately for every delivery day.
    The timestamp contract alone cannot certify historical publication vintages.
    """
    if not isinstance(zone, str) or zone.upper() not in {"DE", "NL"}:
        raise ValueError("Only DE and NL are supported")
    if not isinstance(history_days, int) or not isinstance(min_history_days, int) or not 1 <= min_history_days <= history_days:
        raise ValueError("Require 1 <= min_history_days <= history_days")
    days, boundaries = _daily_grid(covariates, timezone, "covariates")
    prefix = zone.lower()
    columns = [f"{prefix}_wind_generation_fcst", f"{prefix}_solar_generation_fcst", f"{prefix}_residual_load_fcst"]
    values = _finite_columns(covariates, columns, "covariates")
    if (values[:, :2] < 0).any():
        raise ValueError("Generation forecasts must be nonnegative")
    features = np.zeros((len(covariates), len(FEATURE_COLUMNS)), dtype=float)
    daily = []
    for number, day in enumerate(days):
        first = max(0, number - history_days)
        left, right = boundaries[number:number + 2]
        history = values[boundaries[first]:left]
        record = {"delivery_day": str(day), "physical_hours": int(right - left),
                  "history_days": number - first, "history_hours": len(history),
                  "train_delivery_min": str(days[first]) if number else None,
                  "train_delivery_max": str(days[number - 1]) if number else None,
                  "status": "warmup_zero_basis"}
        if number - first >= min_history_days:
            positive_solar = history[history[:, 1] > 0, 1]
            if not len(positive_solar):
                raise ValueError(f"{day}: no positive historical solar scale")
            wind_scale = float(np.quantile(history[:, 0], .75))
            solar_scale = float(np.quantile(positive_solar, .75))
            low, high = np.quantile(history[:, 2], [.5, .9])
            span = high - low
            upper = low + 2 * span
            if (not np.isfinite([wind_scale, solar_scale, low, high, span, upper]).all()
                    or wind_scale <= 0 or solar_scale <= 0 or span <= 0):
                raise ValueError(f"{day}: invalid historical normalization denominators")
            wind, solar, residual = values[left:right].T
            low_wind = 1 - np.minimum(wind, wind_scale) / wind_scale
            low_solar = 1 - np.minimum(solar, solar_scale) / solar_scale
            joint = np.clip(low_wind * low_solar, 0, 1)
            stress = np.clip((np.clip(residual, low, upper) - low) / span, 0, 2)
            features[left:right] = np.column_stack((joint, stress, joint, joint * stress,
                                                     joint * np.maximum(stress - 1, 0)))
            record.update(status="active", wind_q75_gw=wind_scale, solar_positive_q75_gw=solar_scale,
                          residual_q50_gw=float(low), residual_q90_gw=float(high))
        daily.append(record)
    result = pd.DataFrame(features, index=covariates.index.copy(), columns=FEATURE_COLUMNS)
    audit = {"protocol_version": PROTOCOL_VERSION, "zone": zone.upper(), "timezone": timezone,
             "source_columns_read": columns, "source_units": "GW", "history_days": history_days,
             "min_history_days": min_history_days, "normalizations": daily,
             "normalization_window": "[D-history_days,D), complete civil days; D excluded",
             "joint_deficit": "clip(1-W/q75W,0,1)*clip(1-S/q75positiveS,0,1)",
             "residual_stress": "clip((RL-q50RL)/(q90RL-q50RL),0,2)",
             "basis_columns": list(BASIS_COLUMNS), "basis_formula": ["u", "u*g", "u*max(g-1,0)"],
             "prices_or_targets_used": False, "pit_publication_evidence_verified": False,
             "imputation_performed": False}
    return result, audit


def _design(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(BASIS_COLUMNS) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("A finite nonnegative n-by-3 scarcity design is required")
    return x


def fit_nonnegative_premium(x, residuals, *, ridge=0.05):
    """Minimize mean((actual-baseline-X@coef)**2) + ridge*sum(coef**2), coef>=0."""
    x = _design(x)
    residuals = np.asarray(residuals, dtype=float)
    if residuals.shape != (len(x),) or not len(x) or not np.isfinite(residuals).all():
        raise ValueError("Finite aligned historical residual labels required")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Ridge must be finite and strictly positive")
    design = np.vstack((x, np.sqrt(len(x) * ridge) * np.eye(x.shape[1])))
    labels = np.r_[residuals, np.zeros(x.shape[1])]
    coefficients, _ = nnls(design, labels)
    if not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite scarcity fit")
    return coefficients


def predict_premium(x, coefficients, *, cap=None):
    """Nondecreasing positive component at fixed scales/baseline; cap=None is uncapped."""
    x = _design(x)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (len(BASIS_COLUMNS),) or not np.isfinite(coefficients).all() or (coefficients < 0).any():
        raise ValueError("Three finite nonnegative coefficients required")
    if cap is not None and (not np.isfinite(cap) or cap < 0):
        raise ValueError("Cap must be nonnegative finite or None")
    with np.errstate(over="ignore", invalid="ignore"):
        premium = x @ coefficients
    if not np.isfinite(premium).all():
        raise ValueError("Nonfinite premium; refusing an overflowing extrapolation")
    return premium if cap is None else np.minimum(premium, cap)


def run_premium_backtest(backtest, forecast, features, *, timezone, train_days=365,
                         min_train_days=90, ridge=0.05, cap=None):
    """Daily prequential fit on paired sealed final-baseline errors, future excluded.

    Returns scored history only after calibration, the distinct delivery forecast,
    and a daily coefficient audit. All quantiles receive the same price shift.
    Historical price observations for civil D-1 are assumed published by D's
    forecasting origin, as in the parent day-ahead experiment (not newly certified).
    """
    if not isinstance(train_days, int) or not isinstance(min_train_days, int) or not 1 <= min_train_days <= train_days:
        raise ValueError("Require 1 <= min_train_days <= train_days")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Ridge must be finite and strictly positive")
    if cap is not None and (not np.isfinite(cap) or cap < 0):
        raise ValueError("Cap must be nonnegative finite or None")
    days, boundaries = _daily_grid(backtest, timezone, "backtest")
    future_days, _ = _daily_grid(forecast, timezone, "forecast")
    _daily_grid(features, timezone, "features")
    if len(days) <= min_train_days:
        raise ValueError("Backtest must have evaluation days after calibration")
    if len(future_days) != 1 or future_days[0] != days[-1] + timedelta(days=1):
        raise ValueError("Forecast must be one distinct day immediately after backtest")
    if "actual" in forecast and forecast.actual.notna().any():
        raise ValueError("Future actual labels are forbidden")
    baseline = _finite_columns(backtest, BASELINE_COLUMNS, "backtest quantiles")
    future_baseline = _finite_columns(forecast, BASELINE_COLUMNS, "forecast quantiles")
    if (np.diff(baseline, axis=1) < 0).any() or (np.diff(future_baseline, axis=1) < 0).any():
        raise ValueError("Crossed baseline quantiles")
    actual = _finite_columns(backtest, ["actual"], "backtest actual")[:, 0]
    required = backtest.index.append(forecast.index)
    if not required.isin(features.index).all():
        raise ValueError("Missing features for required physical delivery hours")
    feature_values = _finite_columns(features, FEATURE_COLUMNS, "features")
    if (feature_values < 0).any():
        raise ValueError("Scarcity features must be nonnegative")
    all_x = features.loc[required, list(BASIS_COLUMNS)].to_numpy(float)
    residuals = actual - baseline[:, 1]
    premiums = np.zeros(len(required), dtype=float)
    records = []
    for number, day in enumerate(days + future_days):
        first = max(0, number - train_days)
        left = int(boundaries[number])
        right = int(boundaries[number + 1]) if number < len(days) else len(required)
        train_left = int(boundaries[first])
        coefficients = np.zeros(len(BASIS_COLUMNS))
        record = {"delivery_day": str(day), "status": "calibration_only", "is_forecast": number == len(days),
                  "train_delivery_min": str(days[first]) if number else None,
                  "train_delivery_max": str(days[number - 1]) if number else None,
                  "train_days": number - first, "train_hours": left - train_left,
                  "physical_hours": right - left, "ridge": ridge, "cap_eur_mwh": cap}
        if number - first >= min_train_days:
            coefficients = fit_nonnegative_premium(all_x[train_left:left], residuals[train_left:left], ridge=ridge)
            premiums[left:right] = predict_premium(all_x[left:right], coefficients, cap=cap)
            record["status"] = "forecast" if number == len(days) else "evaluated"
        record.update(coefficients=coefficients.tolist(),
                      premium_mean=float(premiums[left:right].mean()),
                      premium_max=float(premiums[left:right].max()),
                      premium_positive_hours=int((premiums[left:right] > 0).sum()))
        records.append(record)

    def augment(frame, shifts):
        result = frame.copy(deep=True)
        result.loc[:, list(FEATURE_COLUMNS)] = features.loc[result.index, list(FEATURE_COLUMNS)]
        result["scarcity_premium"] = shifts
        for q in QUANTILES:
            result["scarcity__" + q] = result["residual_kalman__" + q] + shifts
        _finite_columns(result, ["scarcity__" + q for q in QUANTILES], "shifted quantiles")
        return result

    first_test = int(boundaries[min_train_days])
    return {"backtest": augment(backtest.iloc[first_test:], premiums[first_test:len(backtest)]),
            "forecast": augment(forecast, premiums[len(backtest):]), "daily_audit": records}
