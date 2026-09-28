"""Prospective CPU price and negative-price models for four CWE countries.

This is a new recipe. Historical GPU model scores do not transfer to it.
Every fit uses the 365 complete civil days before its delivery day. Forecast
features are supplied by the separate causal feature builder.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import time

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


PROTOCOL = "nyx_regional_cpu_v1"
ZONES = ("FR", "DE", "BE", "NL")
TIMEZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin",
             "BE": "Europe/Brussels", "NL": "Europe/Amsterdam"}
CANDIDATES = ("absolute", "residual_prior_day_mean", "blend50")
NEGATIVE_METHODS = ("p_negative", "p_negative_raw", "history_frequency")
PRICE_PARAMETERS = {"loss_function": "RMSE", "iterations": 250, "depth": 6,
    "learning_rate": .04, "l2_leaf_reg": 10., "random_seed": 20260928,
    "task_type": "CPU", "nan_mode": "Min", "verbose": False,
    "allow_writing_files": False, "use_best_model": False}
NEGATIVE_PARAMETERS = {"loss_function": "Logloss", "iterations": 120,
    "depth": 4, "learning_rate": .05, "l2_leaf_reg": 3.,
    "random_seed": 20260928, "task_type": "CPU", "nan_mode": "Min",
    "verbose": False, "allow_writing_files": False, "use_best_model": False}
CALIBRATION_DAYS = 28


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def civil_day(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    require(pd.notna(stamp) and stamp.tzinfo is None
            and stamp == stamp.normalize() and str(stamp.date()) == str(value),
            "Exact YYYY-MM-DD civil date required")
    return stamp


def grid(first_day: str | pd.Timestamp, stop_day: str | pd.Timestamp, zone: str) -> pd.DatetimeIndex:
    require(zone in ZONES, "Unsupported country")
    return pd.date_range(str(pd.Timestamp(first_day).date()), str(pd.Timestamp(stop_day).date()),
                         tz=TIMEZONES[zone], freq="h", inclusive="left").tz_convert("UTC")


def frame_hash(frame: pd.DataFrame | pd.Series) -> str:
    names = list(frame.columns) if isinstance(frame, pd.DataFrame) else [frame.name]
    require(isinstance(frame.index, pd.DatetimeIndex) and str(frame.index.tz) == "UTC",
            "Hash requires UTC index")
    values = np.asarray(frame, dtype="<f8")
    raw = frame.index.as_unit("ns").asi8.astype("<i8", copy=False).tobytes()
    raw += json.dumps(names, ensure_ascii=False).encode("utf-8")
    raw += values.tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def validate_inputs(features: pd.DataFrame, actual: pd.Series, *, zone: str,
                    delivery_day: str, stop_day: str | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Timestamp]:
    day = civil_day(delivery_day)
    stop = civil_day(stop_day) if stop_day is not None else day + pd.Timedelta(days=1)
    require(zone in ZONES, "Unsupported country")
    require(day < stop <= day + pd.Timedelta(days=7), "One to seven future civil days required")
    first = day - pd.Timedelta(days=365)
    train_index, future_index = grid(first, day, zone), grid(day, stop, zone)
    expected = train_index.append(future_index)
    require(isinstance(features, pd.DataFrame) and features.index.equals(expected)
            and isinstance(features.index, pd.DatetimeIndex) and str(features.index.tz) == "UTC"
            and features.index.is_unique and not features.index.hasnans
            and features.index.is_monotonic_increasing and features.columns.is_unique
            and len(features.columns) > 0
            and all(isinstance(name, str) and name for name in features.columns),
            "Features require complete ordered UTC train and future grids")
    require(not any(any(term in name.lower() for term in
                    ("storm", "actual", "observed", "target", "label")) for name in features),
            "Labels and Storm cannot be features")
    require(all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes),
            "Numeric feature schema required")
    x = features.astype(float)
    require(not np.isinf(x.to_numpy()).any(), "Infinite features forbidden")
    flags = [name for name in x if name.endswith("__available")]
    require(not flags or x[flags].isin([0., 1.]).all().all(),
            "Availability indicators must be binary")
    require(isinstance(actual, pd.Series) and actual.index.equals(train_index)
            and str(actual.index.tz) == "UTC" and "storm" not in str(actual.name).lower(),
            "Actual prices must be exactly the preceding 365 civil days")
    y = pd.to_numeric(actual, errors="raise").astype(float)
    require(np.isfinite(y.to_numpy()).all(), "Complete finite historical prices required")
    return x.loc[train_index], y, x.loc[future_index], day


def prior_day_mean(prices: pd.Series, index: pd.DatetimeIndex, *, zone: str) -> pd.Series:
    """Use the previous complete local day's observed mean, never today's label."""
    require(isinstance(prices, pd.Series) and isinstance(prices.index, pd.DatetimeIndex)
            and str(prices.index.tz) == "UTC" and prices.index.is_unique
            and prices.index.is_monotonic_increasing, "Ordered UTC price history required")
    local = prices.index.tz_convert(TIMEZONES[zone])
    daily = prices.groupby(local.date).agg(["mean", "count"])
    days = index.tz_convert(TIMEZONES[zone]).date
    previous = [day - timedelta(days=1) for day in days]
    means = daily["mean"].reindex(previous).to_numpy(float)
    counts = daily["count"].reindex(previous).to_numpy(float)
    expected = np.asarray([len(grid(day, day + timedelta(days=1), zone)) for day in previous])
    require(np.isfinite(means).all() and np.array_equal(counts, expected),
            "Previous civil-day price means require complete past days")
    return pd.Series(means, index=index, name="prior_day_mean")


@dataclass
class PriceResult:
    predictions: pd.DataFrame
    models: dict[str, CatBoostRegressor]
    audit: dict


def fit_price_candidates(features: pd.DataFrame, actual: pd.Series, prices: pd.Series,
                         *, zone: str, delivery_day: str, stop_day: str | None = None,
                         selected: str | None = None, threads: int = 2) -> PriceResult:
    require(type(threads) is int and 1 <= threads <= 8, "1-8 CPU threads required")
    require(selected is None or selected in CANDIDATES, "Unknown selected price candidate")
    train_x, train_y, future_x, day = validate_inputs(features, actual, zone=zone,
                                                       delivery_day=delivery_day, stop_day=stop_day)
    base = prior_day_mean(prices, train_x.index.append(future_x.index), zone=zone)
    train_base, future_base = base.loc[train_x.index], base.loc[future_x.index]
    params = {**PRICE_PARAMETERS, "thread_count": threads}
    started = time.perf_counter()
    models = {}
    points = {}
    if selected in (None, "absolute", "blend50"):
        absolute = CatBoostRegressor(**params)
        absolute.fit(train_x, train_y)
        models["absolute"] = absolute
        points["absolute"] = np.asarray(absolute.predict(future_x), dtype=float)
    if selected in (None, "residual_prior_day_mean", "blend50"):
        residual = CatBoostRegressor(**params)
        residual.fit(train_x, train_y.to_numpy(float) - train_base.to_numpy(float))
        models["residual_prior_day_mean"] = residual
        points["residual_prior_day_mean"] = (future_base.to_numpy(float)
            + np.asarray(residual.predict(future_x), dtype=float))
    if "absolute" in points and "residual_prior_day_mean" in points:
        points["blend50"] = (points["absolute"] + points["residual_prior_day_mean"]) / 2.
    require(all(np.isfinite(point).all() for point in points.values()), "Nonfinite price forecasts")
    if selected is not None:
        points = {selected: points[selected]}
    predictions = pd.DataFrame(points, index=future_x.index)
    audit = {"protocol": PROTOCOL, "zone": zone, "delivery_day": str(day.date()),
        "history_first_day": str((day - pd.Timedelta(days=365)).date()),
        "history_last_day": str((day - pd.Timedelta(days=1)).date()),
        "history_hours": len(train_x), "forecast_hours": len(future_x),
        "forecast_stop_day_exclusive": str(future_x.index[-1].tz_convert(TIMEZONES[zone]).date()
                                               + timedelta(days=1)),
        "feature_columns": list(train_x.columns), "feature_sha256": frame_hash(train_x),
        "future_feature_sha256": frame_hash(future_x), "actual_sha256": frame_hash(train_y),
        "prior_day_mean_sha256": frame_hash(base), "parameters": params,
        "candidate_formulas": {"absolute": "CatBoost(actual)",
            "residual_prior_day_mean": "prior_day_mean + CatBoost(actual - prior_day_mean)",
            "blend50": "(absolute + residual_prior_day_mean) / 2"},
        "current_or_future_actual_used": False, "Storm_used": False,
        "prediction_sha256": frame_hash(predictions),
        "fit_seconds": time.perf_counter() - started,
        "tree_counts": {name: int(estimator.tree_count_) for name, estimator in models.items()}}
    return PriceResult(predictions, models, audit)


@dataclass
class NegativeResult:
    predictions: pd.DataFrame
    model: CatBoostClassifier | None
    audit: dict


def fit_negative_price(features: pd.DataFrame, actual: pd.Series, *, zone: str,
                       delivery_day: str, stop_day: str | None = None,
                       threads: int = 2) -> NegativeResult:
    require(type(threads) is int and 1 <= threads <= 8, "1-8 CPU threads required")
    train_x, train_y, future_x, day = validate_inputs(features, actual, zone=zone,
                                                       delivery_day=delivery_day, stop_day=stop_day)
    cut_day = day - pd.Timedelta(days=CALIBRATION_DAYS)
    fit_index, cal_index = grid(day - pd.Timedelta(days=365), cut_day, zone), grid(cut_day, day, zone)
    target = train_y.lt(0.).astype(int)
    fit_x, fit_y, cal_x, cal_y = train_x.loc[fit_index], target.loc[fit_index], train_x.loc[cal_index], target.loc[cal_index]
    started = time.perf_counter()
    fallback_reason = ("one_class" if fit_y.nunique() < 2 else
                       "constant_features" if not (fit_x.nunique(dropna=False) > 1).any() else None)
    model = None
    slope = intercept = None
    if fallback_reason:
        raw = np.full(len(future_x), float(target.mean()))
        final = raw.copy()
        calibration = "frequency_fallback"
    else:
        model = CatBoostClassifier(**{**NEGATIVE_PARAMETERS, "thread_count": threads})
        model.fit(fit_x, fit_y)
        raw = np.asarray(model.predict_proba(future_x))[:, 1]
        if int(cal_y.sum()) >= 10 and int((1 - cal_y).sum()) >= 10:
            margins = np.asarray(model.predict(cal_x, prediction_type="RawFormulaVal"), dtype=float)
            calibrator = LogisticRegression(C=1., solver="lbfgs", max_iter=1000,
                                            random_state=20260928)
            calibrator.fit(margins.reshape(-1, 1), cal_y)
            require(int(calibrator.n_iter_[0]) < 1000, "Negative-price calibration did not converge")
            slope = float(calibrator.coef_[0, 0])
            intercept = float(calibrator.intercept_[0])
            future_margin = np.asarray(model.predict(future_x, prediction_type="RawFormulaVal"), dtype=float)
            final = expit(slope * future_margin + intercept)
            calibration = "platt"
        else:
            final = raw.copy()
            calibration = "raw_insufficient_calibration_classes"
    require(np.isfinite(raw).all() and np.isfinite(final).all()
            and ((raw >= 0.) & (raw <= 1.)).all()
            and ((final >= 0.) & (final <= 1.)).all(), "Invalid negative-price probability")
    result = pd.DataFrame({"p_negative_raw": raw, "p_negative": final,
                           "is_negative_predicted": final >= .5}, index=future_x.index)
    audit = {"protocol": PROTOCOL, "zone": zone, "delivery_day": str(day.date()),
        "event": "actual hourly price < 0", "alert_threshold": .5,
        "fit_days": 337, "calibration_days": CALIBRATION_DAYS,
        "fit_hours": len(fit_x), "calibration_hours": len(cal_x),
        "fit_negative_count": int(fit_y.sum()), "calibration_negative_count": int(cal_y.sum()),
        "fallback_reason": fallback_reason, "calibration_status": calibration,
        "calibration_slope": slope, "calibration_intercept": intercept,
        "parameters": {**NEGATIVE_PARAMETERS, "thread_count": threads},
        "history_features_sha256": frame_hash(train_x),
        "history_actual_sha256": frame_hash(train_y),
        "future_features_sha256": frame_hash(future_x),
        "probabilities_sha256": frame_hash(result[["p_negative_raw", "p_negative", "is_negative_predicted"]]),
        "current_or_future_actual_used": False, "Storm_used": False,
        "fit_seconds": time.perf_counter() - started,
        "tree_count": 0 if model is None else int(model.tree_count_)}
    return NegativeResult(result, model, audit)


def fit_predict_block(features_by_zone: dict[str, pd.DataFrame],
                      actual_by_zone: dict[str, pd.Series],
                      price_history_by_zone: dict[str, pd.Series], *,
                      origin_day: str, stop_day: str, candidate: str | dict[str, str] | None = None,
                      threads: int = 2) -> tuple[dict[str, pd.DataFrame], dict]:
    """Fit chronological price candidates for one origin across CWE.

    ``candidate=None`` emits all three predefined candidates for backtesting.
    A named candidate or a country mapping emits only the selected column.
    Storm and current/future observations are never accepted as model inputs.
    """
    require(set(features_by_zone) == set(actual_by_zone)
            == set(price_history_by_zone) == set(ZONES), "All four CWE countries required")
    if isinstance(candidate, dict):
        require(set(candidate) == set(ZONES)
                and all(value in CANDIDATES for value in candidate.values()),
                "One predefined candidate per country required")
    else:
        require(candidate is None or candidate in CANDIDATES, "Unknown candidate")
    outputs = {}
    audits = {}
    for zone in ZONES:
        selected = candidate[zone] if isinstance(candidate, dict) else candidate
        fitted = fit_price_candidates(features_by_zone[zone], actual_by_zone[zone],
            price_history_by_zone[zone], zone=zone, delivery_day=origin_day,
            stop_day=stop_day, selected=selected, threads=threads)
        outputs[zone], audits[zone] = fitted.predictions, fitted.audit
    return outputs, {"protocol": PROTOCOL, "origin_day": origin_day,
                     "stop_day_exclusive": stop_day, "candidate": candidate,
                     "countries": audits, "Storm_used": False}


def fit_predict_negative_block(features_by_zone: dict[str, pd.DataFrame],
                               actual_by_zone: dict[str, pd.Series], *,
                               origin_day: str, stop_day: str,
                               threads: int = 2) -> tuple[dict[str, pd.DataFrame], dict]:
    """Fit and calibrate hourly P(price < 0) with strictly earlier labels."""
    require(set(features_by_zone) == set(actual_by_zone) == set(ZONES),
            "All four CWE countries required")
    outputs = {}
    audits = {}
    for zone in ZONES:
        fitted = fit_negative_price(features_by_zone[zone], actual_by_zone[zone],
            zone=zone, delivery_day=origin_day, stop_day=stop_day, threads=threads)
        outputs[zone], audits[zone] = fitted.predictions, fitted.audit
    return outputs, {"protocol": PROTOCOL, "origin_day": origin_day,
                     "stop_day_exclusive": stop_day, "countries": audits,
                     "Storm_used": False}
