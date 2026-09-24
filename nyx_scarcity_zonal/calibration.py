"""Regularised country offsets on an already fitted shared Platt probability.

There is no independent country slope, class weighting, severity fitting or
amplitude selection. Each offset minimises a SUM of binary log losses, with
physical-hour weights 24 / (23, 24 or 25) and penalty * delta**2 / 2. Thus every
complete civil country-day contributes 24 loss units; the penalty is not scaled
by the number of observations. This is a loss normalisation, not a claim that
hours or neighbouring countries are statistically independent.

Input probabilities must be strictly inside (0, 1). Any upstream numerical
clipping belongs to the caller and must be audited there, not hidden here.
All supplied labels must already be known at the cutoff; invalid/future rows
raise rather than being silently discarded. This module does not perform I/O.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit


ZONES = ("BE", "DE", "FR", "NL")
TIMEZONE = "Europe/Paris"


class ZonalCalibrationError(ValueError):
    """The fixed chronological calibration contract is invalid."""


def _date(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ZonalCalibrationError(f"{name}: an explicit YYYY-MM-DD string is required.")
    try:
        parsed = pd.Timestamp(value)
        valid = not pd.isna(parsed) and parsed.tzinfo is None and parsed.strftime("%Y-%m-%d") == value
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ZonalCalibrationError(f"{name}: an explicit YYYY-MM-DD string is required.")
    return value


def _aware(value: Any, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ZonalCalibrationError(f"{name}: an explicit aware timestamp is required.") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ZonalCalibrationError(f"{name}: an explicit aware timestamp is required.")
    return timestamp.tz_convert("UTC")


def _timestamps(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise ZonalCalibrationError(f"{name}: missing timestamps are forbidden.")
    try:
        if any(pd.Timestamp(value).tzinfo is None for value in values):
            raise ZonalCalibrationError(f"{name}: timezone-naive timestamps are forbidden.")
        return pd.to_datetime(values, utc=True, format="mixed")
    except (ValueError, TypeError) as exc:
        raise ZonalCalibrationError(f"{name}: invalid timestamps.") from exc


def _probabilities(values, *, size: int | None = None, index=None) -> np.ndarray:
    if isinstance(values, pd.Series) and index is not None and not values.index.equals(index):
        raise ZonalCalibrationError("Probability Series must preserve calibration row-index alignment.")
    array = np.asarray(values)
    if (array.ndim != 1 or (size is not None and len(array) != size)
            or array.dtype.kind not in "fiu" or array.dtype.kind == "b"):
        raise ZonalCalibrationError("One-dimensional aligned numeric probabilities are required.")
    result = array.astype(float)
    if not np.isfinite(result).all() or np.any(result <= 0) or np.any(result >= 1):
        raise ZonalCalibrationError("Probabilities must be finite and strictly inside (0, 1); no hidden clipping.")
    return result


def _events(values, *, size: int, index) -> np.ndarray:
    if isinstance(values, pd.Series) and not values.index.equals(index):
        raise ZonalCalibrationError("Event Series must preserve calibration row-index alignment.")
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != size:
        raise ZonalCalibrationError("One-dimensional aligned binary events are required.")
    if array.dtype.kind not in "bfiu" or not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise ZonalCalibrationError("Events must be explicit finite booleans or binary 0/1 values.")
    return array.astype(float)


def _positive_number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ZonalCalibrationError(f"{name}: a finite strictly positive number is required.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ZonalCalibrationError(f"{name}: a finite strictly positive number is required.")
    return number


def _validated_frame(calibration_frame: pd.DataFrame, *, cutoff: pd.Timestamp, current_day: str):
    required = {"zone", "timestamp_utc", "label_available_at_utc"}
    if (not isinstance(calibration_frame, pd.DataFrame) or calibration_frame.columns.has_duplicates
            or not required.issubset(calibration_frame)):
        raise ZonalCalibrationError("Calibration requires unique zone, timestamp_utc and label_available_at_utc columns.")
    frame = calibration_frame[[*sorted(required), *[key for key in ("_day", "forecast_origin_utc") if key in calibration_frame]]].copy()
    if not frame.zone.isin(ZONES).all():
        raise ZonalCalibrationError("Only explicit BE/DE/FR/NL country codes are accepted.")
    frame["timestamp_utc"] = _timestamps(frame.timestamp_utc, "timestamp_utc")
    frame["label_available_at_utc"] = _timestamps(frame.label_available_at_utc, "label_available_at_utc")
    if (frame.duplicated(["zone", "timestamp_utc"]).any()
            or not frame.timestamp_utc.eq(frame.timestamp_utc.dt.floor("h")).all()):
        raise ZonalCalibrationError("Unique physical hourly country/delivery identities are required.")
    civil = frame.timestamp_utc.dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize()
    day = civil.dt.strftime("%Y-%m-%d")
    if "_day" in frame and not frame._day.eq(day).all():
        raise ZonalCalibrationError("Supplied civil-day labels disagree with physical timestamps.")
    frame["_day"] = day
    if (day.ge(current_day).any() or frame.label_available_at_utc.gt(cutoff).any()):
        raise ZonalCalibrationError("Current/future deliveries or labels published after cutoff are forbidden, not filtered.")
    own_origin = (civil - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).dt.tz_localize(TIMEZONE).dt.tz_convert("UTC")
    if "forecast_origin_utc" in frame:
        frame["forecast_origin_utc"] = _timestamps(frame.forecast_origin_utc, "forecast_origin_utc")
        if not frame.forecast_origin_utc.eq(own_origin).all():
            raise ZonalCalibrationError("Calibration forecast origins must preserve D-1 08:00 civil.")
    if frame.label_available_at_utc.le(own_origin).any():
        raise ZonalCalibrationError("Calibration labels must become known strictly after their own forecast origin.")
    # Work positionally after validating any external Series against the caller's
    # original index; no sorting may realign probabilities with another target.
    frame = frame.reset_index(drop=True)
    weights = np.empty(len(frame), dtype=float)
    hours = np.empty(len(frame), dtype=int)
    for (zone, date), positions in frame.groupby(["zone", "_day"], sort=True).indices.items():
        start = pd.Timestamp(date).tz_localize(TIMEZONE)
        finish = (pd.Timestamp(date) + pd.Timedelta(days=1)).tz_localize(TIMEZONE)
        expected = pd.date_range(start, finish, freq="h", inclusive="left").tz_convert("UTC")
        actual = pd.DatetimeIndex(frame.iloc[positions].timestamp_utc).sort_values()
        if not actual.equals(expected):
            raise ZonalCalibrationError(f"{zone}/{date}: incomplete physical civil day; expected {len(expected)} hours, received {len(actual)}.")
        weights[positions] = 24. / len(expected)
        hours[positions] = len(expected)
    frame["_weight"], frame["_physical_day_hours"] = weights, hours
    return frame


def _objective(delta: float, margin: np.ndarray, events: np.ndarray, weights: np.ndarray, penalty: float) -> float:
    signed_margin = np.where(events == 1., -(margin + delta), margin + delta)
    return float(np.dot(weights, np.logaddexp(0., signed_margin)) + penalty * delta * delta / 2.)


def fit_zone_offsets(calibration_frame: pd.DataFrame, shared_probabilities, events, *, cutoff,
                     current_day: str, penalty: float = 1.0, max_abs: float = 3.0) -> tuple[dict[str, float], dict]:
    """Fit four bounded offsets; an absent country explicitly receives zero.

    Every supplied observation is validated and used. Countries with zero
    positive or zero negative events remain valid: the fixed L2 penalty and
    hard bound give a finite estimate, not an empirical probability of 0 or 1.
    """
    current_day = _date(current_day, "current_day")
    cutoff = _aware(cutoff, "cutoff")
    expected_cutoff = (pd.Timestamp(current_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(TIMEZONE).tz_convert("UTC")
    if cutoff != expected_cutoff:
        raise ZonalCalibrationError("Current cutoff must equal D-1 08:00 Europe/Paris civil.")
    penalty, max_abs = _positive_number(penalty, "penalty"), _positive_number(max_abs, "max_abs")
    if not math.isfinite(penalty * max_abs * max_abs):
        raise ZonalCalibrationError("Penalty and bound overflow finite arithmetic.")
    if not isinstance(calibration_frame, pd.DataFrame):
        raise ZonalCalibrationError("Calibration input must be a DataFrame.")
    probabilities = _probabilities(shared_probabilities, size=len(calibration_frame), index=calibration_frame.index)
    binary = _events(events, size=len(calibration_frame), index=calibration_frame.index)
    frame = _validated_frame(calibration_frame, cutoff=cutoff, current_day=current_day)
    margin = logit(probabilities)
    offsets, zones = {}, {}
    for zone in ZONES:
        mask = frame.zone.eq(zone).to_numpy()
        selected = frame.loc[mask]
        if selected.empty:
            offsets[zone] = 0.
            zones[zone] = {"status": "no_calibration_rows_identity", "offset": 0., "rows": 0,
                           "civil_days": 0, "positive_events": 0, "negative_events": 0,
                           "effective_loss_weight": 0., "at_bound": False}
            continue
        m, y, w = margin[mask], binary[mask], selected._weight.to_numpy(float)
        def gradient(delta):
            return float(np.dot(w, expit(m + delta) - y) + penalty * delta)
        lower, upper = gradient(-max_abs), gradient(max_abs)
        if lower >= 0:
            delta, status = -max_abs, "lower_bound_optimum"
        elif upper <= 0:
            delta, status = max_abs, "upper_bound_optimum"
        else:
            delta, status = float(brentq(gradient, -max_abs, max_abs, xtol=1e-12, rtol=1e-12)), "interior_optimum"
        adjusted = expit(m + delta)
        before, after = _objective(0., m, y, w, penalty), _objective(delta, m, y, w, penalty)
        if not math.isfinite(after) or after > before + 1e-9 * max(1., abs(before)):
            raise ZonalCalibrationError("Offset optimisation failed to minimise the declared penalised objective.")
        offsets[zone] = float(delta)
        counts = selected.groupby(["_day", "_physical_day_hours"]).size().reset_index(name="rows")
        zones[zone] = {
            "status": status, "offset": float(delta), "rows": len(selected), "civil_days": int(selected._day.nunique()),
            "start_day": str(selected._day.min()), "end_day": str(selected._day.max()),
            "positive_events": int(y.sum()), "negative_events": int(len(y) - y.sum()),
            "positive_event_days": int(selected.loc[y.astype(bool), "_day"].nunique()),
            "physical_day_counts": {str(int(hours)): int(number) for hours, number in counts._physical_day_hours.value_counts().items()},
            "effective_loss_weight": float(w.sum()), "weighted_positive_events": float(np.dot(w, y)),
            "mean_shared_probability": float(probabilities[mask].mean()), "mean_adjusted_probability": float(adjusted.mean()),
            "weighted_expected_events_before": float(np.dot(w, probabilities[mask])),
            "weighted_expected_events_after": float(np.dot(w, adjusted)),
            "objective_before": before, "objective_after": after, "penalty_component_after": penalty * delta * delta / 2.,
            "gradient_at_solution": gradient(delta), "at_bound": bool(abs(delta) >= max_abs - 1e-10),
            "maximum_label_available_at_utc": selected.label_available_at_utc.max().isoformat(),
        }
    audit = {
        "schema_version": 1, "method": "bounded_l2_country_intercept_on_fixed_shared_platt_logit",
        "formula": "p_zone = sigmoid(logit(p_shared) + offset_zone)",
        "objective": "sum_h (24 / physical_hours_in_civil_day) * binary_log_loss(y_h, p_zone_h) + penalty * offset_zone**2 / 2",
        "first_order_equation": "sum_h w_h * (sigmoid(logit(p_shared_h)+offset_zone)-y_h) + penalty*offset_zone = 0, subject to bounds",
        "penalty": penalty, "max_abs_offset": max_abs,
        "penalty_scale": "fixed penalty against SUM loss, not mean loss; a complete country-day has 24 loss units",
        "timezone": TIMEZONE, "cutoff_utc": cutoff.isoformat(), "current_day": current_day,
        "rows": len(frame), "rows_filtered": 0, "probability_clipping_performed": False,
        "input_probability_domain": "strictly inside (0,1); any caller clipping must have its own audit",
        "future_labels_rejected": True, "current_delivery_labels_rejected": True,
        "complete_physical_civil_days_required": True, "all_country_offsets_explicit": True,
        "empty_country_policy": "offset 0 with no_calibration_rows_identity status",
        "class_reweighting": False, "independent_country_slopes": False, "severity_model_changed": False,
        "amplitude_selected": False, "independent_hour_assumption": False, "confidence_intervals_provided": False,
        "interpretation": "Regularised conditional probability adjustment, not proof of causality or future calibration/non-regression.",
        "zones": zones,
    }
    return offsets, audit


def apply_zone_offsets(probabilities, zones, offsets: Mapping[str, float]) -> np.ndarray:
    """Apply explicit saved offsets without data-dependent fitting or fallback."""
    values = _probabilities(probabilities)
    countries = np.asarray(zones)
    if countries.ndim != 1 or len(countries) != len(values) or not np.isin(countries, ZONES).all():
        raise ZonalCalibrationError("Aligned BE/DE/FR/NL country identities are required.")
    if isinstance(probabilities, pd.Series) and isinstance(zones, pd.Series) and not probabilities.index.equals(zones.index):
        raise ZonalCalibrationError("Probability and country Series must preserve index alignment.")
    if not isinstance(offsets, Mapping) or set(offsets) != set(ZONES):
        raise ZonalCalibrationError("All four country offsets must be explicit; missing or unknown keys cannot silently fall back.")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating, np.integer))
           or not math.isfinite(float(value)) for value in offsets.values()):
        raise ZonalCalibrationError("Every saved country offset must be finite and numeric.")
    delta = np.asarray([float(offsets[str(zone)]) for zone in countries])
    result = expit(logit(values) + delta)
    if not np.isfinite(result).all():
        raise ZonalCalibrationError("Non-finite adjusted probabilities.")
    return result


__all__ = ["ZonalCalibrationError", "fit_zone_offsets", "apply_zone_offsets"]
