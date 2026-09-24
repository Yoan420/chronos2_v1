"""Isolated, retrospective DE/NL scarcity-regime residual distributions.

This module performs no filesystem or network access and has no production
integration. Inputs are frozen forecasts and strictly earlier observed residuals
against the final interaction-40 P50. The caller owns source SHA validation,
weekly refit scheduling, and the availability/publication audit of every input.

The gate estimates P(residual > 50); two conditional quantile experts describe
the normal and positive-spike residual distributions. Their CDFs are mixed and
inverted: a probability-weighted mean of expert medians is NOT called a P50.
Returned residual quantiles must each be added to the SAME frozen baseline P50,
not to the baseline's respective P10/P50/P90. This is a new conditional residual
distribution, not a convolution with the existing baseline uncertainty.

No return value certifies PIT publication history or prospective calibration.
In particular, inspection of 22 September informed this hypothesis; that event
is a diagnostic, not an untouched test. Very sparse spikes use an explicit
empirical fallback, and tree/empirical tails cannot extrapolate unseen extremes.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "solar_wind_scarcity_regime_v1"
SUPPORTED_ZONES = ("DE", "NL")
FEATURE_HISTORY_DAYS = 365
FEATURE_WARMUP_DAYS = 14
TRAINING_WARMUP_DAYS = 90
SPIKE_THRESHOLD = 50.0
CALIBRATION_DAYS = 14
MINIMUM_GATE_SPIKES = 8
MINIMUM_GATE_NORMAL = 32
MINIMUM_EXPERT_ROWS = {"normal": 128, "spike": 32}
OUTPUT_PROBABILITIES = np.array([0.10, 0.50, 0.90], dtype=float)
EXPERT_PROBABILITIES = np.array(
    [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0],
    dtype=float,
)


def _validated_source(frame: pd.DataFrame, zone: str, timezone: str) -> np.ndarray:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{zone}: a nonempty source frame is required")
    idx = frame.index
    if (not isinstance(idx, pd.DatetimeIndex) or str(idx.tz) != "UTC"
            or idx.hasnans or not idx.is_unique or not idx.is_monotonic_increasing
            or not idx.equals(idx.floor("h"))):
        raise ValueError(f"{zone}: ordered unique hourly UTC timestamps are required")
    if not frame.columns.is_unique:
        raise ValueError(f"{zone}: duplicate source columns")
    columns = [f"{zone.lower()}_{suffix}" for suffix in
               ("wind_generation_fcst", "solar_generation_fcst", "residual_load_fcst")]
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"{zone}: missing frozen W/S/RL forecast columns")
    values = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :2] < 0).any():
        raise ValueError(f"{zone}: finite forecasts and nonnegative W/S required; no imputation")
    local_days = idx.tz_convert(timezone).date
    start = pd.Timestamp(local_days[0]).tz_localize(timezone)
    end = pd.Timestamp(local_days[-1] + timedelta(days=1)).tz_localize(timezone)
    expected = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    if not idx.equals(expected):
        raise ValueError(f"{zone}: complete consecutive civil days, including DST hours, required")
    return values


def build_features(
    covariates: Mapping[str, pd.DataFrame], timezone: str = "Europe/Berlin"
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Build aligned country frames from frozen, same-origin DE/NL forecasts.

    Each mapping value needs only its own three lowercase-prefixed forecast
    columns. Both UTC indexes must match exactly. Normalizers for civil day D
    use complete forecast days in [D-365,D). First 14 days have NaN features and
    MUST be explicitly excluded by the caller; there is no imputation. Ramps,
    leads and daily peaks use only the forecast profile for that SAME delivery
    day. Those future delivery hours are known at that origin by input contract;
    no profile from the next civil day is consulted.
    """
    ZoneInfo(timezone)
    if set(covariates) != set(SUPPORTED_ZONES):
        raise ValueError("Exactly DE and NL frozen source frames are required")
    source = {zone: _validated_source(covariates[zone], zone, timezone)
              for zone in SUPPORTED_ZONES}
    index = covariates["DE"].index
    if not index.equals(covariates["NL"].index):
        raise ValueError("DE and NL frozen forecasts must have identical UTC indexes")
    local = index.tz_convert(timezone)
    local_days = local.date
    days = list(dict.fromkeys(local_days))
    boundaries = np.r_[0, np.flatnonzero(local_days[1:] != local_days[:-1]) + 1, len(index)]
    component_names = (
        "wind_gw", "solar_gw", "residual_load_gw", "wind_ratio", "solar_ratio",
        "low_wind", "low_solar", "joint_deficit", "residual_stress",
        "deficit_stress", "rl_ramp_previous", "rl_ramp_next1", "rl_ramp_next2",
        "wind_drop_next2", "solar_drop_next2", "stress_next1", "stress_next2",
        "daily_peak_residual_stress", "daily_peak_deficit_stress",
    )
    components = {
        zone: pd.DataFrame(np.nan, index=index.copy(), columns=component_names)
        for zone in SUPPORTED_ZONES
    }
    normalizations: dict[str, list[dict]] = {zone: [] for zone in SUPPORTED_ZONES}
    for day_number, day in enumerate(days):
        left, right = int(boundaries[day_number]), int(boundaries[day_number + 1])
        first_day = max(0, day_number - FEATURE_HISTORY_DAYS)
        history_left = int(boundaries[first_day])
        for zone in SUPPORTED_ZONES:
            record = {
                "delivery_day": day.isoformat(), "physical_hours": right - left,
                "history_complete_days": day_number - first_day,
                "history_last_day": days[day_number - 1].isoformat() if day_number else None,
                "history_first_day": days[first_day].isoformat() if day_number else None,
                "status": "excluded_feature_warmup",
            }
            if day_number - first_day < FEATURE_WARMUP_DAYS:
                normalizations[zone].append(record)
                continue
            history = source[zone][history_left:left]
            positive_solar = history[history[:, 1] > 0, 1]
            if not len(positive_solar):
                raise ValueError(f"{zone} {day}: no positive historical solar normalization")
            wind_scale = float(np.quantile(history[:, 0], .75))
            solar_scale = float(np.quantile(positive_solar, .75))
            r50, r90 = np.quantile(history[:, 2], [.5, .9])
            span = float(r90 - r50)
            if (not np.isfinite([wind_scale, solar_scale, r50, r90, span]).all()
                    or wind_scale <= 0 or solar_scale <= 0 or span <= 0):
                raise ValueError(f"{zone} {day}: invalid historical normalization denominator")
            current = source[zone][left:right]
            wind, solar, residual = current.T
            low_wind = np.clip(1.0 - wind / wind_scale, 0.0, 1.0)
            low_solar = np.clip(1.0 - solar / solar_scale, 0.0, 1.0)
            deficit = low_wind * low_solar
            # Neither upper nor lower residual stress is clipped. In particular,
            # q90 is a scale reference, not a severity saturation threshold.
            stress = (residual - r50) / span
            deficit_stress = deficit * np.maximum(stress, 0.0)
            position = np.arange(right - left)
            previous = np.maximum(position - 1, 0)
            next1 = np.minimum(position + 1, right - left - 1)
            next2 = np.minimum(position + 2, right - left - 1)
            matrix = np.column_stack([
                wind, solar, residual, wind / wind_scale, solar / solar_scale,
                low_wind, low_solar, deficit, stress, deficit_stress,
                (residual - residual[previous]) / span,
                (residual[next1] - residual) / span,
                (residual[next2] - residual) / span,
                (wind - wind[next2]) / wind_scale,
                (solar - solar[next2]) / solar_scale,
                stress[next1], stress[next2],
                np.full(len(position), stress.max()),
                np.full(len(position), deficit_stress.max()),
            ])
            if not np.isfinite(matrix).all():
                raise ValueError(f"{zone} {day}: nonfinite normalized feature; no clipping/imputation")
            components[zone].iloc[left:right] = matrix
            record.update(status="ready", wind_q75_gw=wind_scale,
                          solar_positive_q75_gw=solar_scale, residual_q50_gw=float(r50),
                          residual_q90_gw=float(r90), residual_span_gw=span)
            normalizations[zone].append(record)
    result = {}
    for zone in SUPPORTED_ZONES:
        other = "NL" if zone == "DE" else "DE"
        own, neighbor = components[zone], components[other]
        frame = pd.concat([own.add_prefix("own_"), neighbor.add_prefix("other_")], axis=1)
        frame["regional_max_residual_stress"] = np.maximum(
            own["residual_stress"], neighbor["residual_stress"])
        frame["regional_mean_residual_stress"] = (
            own["residual_stress"] + neighbor["residual_stress"]) / 2
        frame["regional_joint_deficit"] = own["joint_deficit"] * neighbor["joint_deficit"]
        frame["country_is_nl"] = float(zone == "NL")
        frame["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
        frame["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
        frame["weekday_sin"] = np.sin(2 * np.pi * local.dayofweek / 7)
        frame["weekday_cos"] = np.cos(2 * np.pi * local.dayofweek / 7)
        frame["annual_sin"] = np.sin(2 * np.pi * (local.dayofyear - 1) / 365.25)
        frame["annual_cos"] = np.cos(2 * np.pi * (local.dayofyear - 1) / 365.25)
        # Warmup is explicitly unusable, including otherwise-known calendar data.
        frame.loc[own["wind_gw"].isna(), :] = np.nan
        result[zone] = frame.loc[:, sorted(frame.columns)]
    return result, {
        "protocol_version": PROTOCOL_VERSION, "timezone": timezone,
        "rows_per_zone": len(index), "feature_names": list(result["DE"].columns),
        "normalization_history_days": FEATURE_HISTORY_DAYS,
        "excluded_feature_warmup_days": FEATURE_WARMUP_DAYS,
        "normalization_rule": "complete civil forecast days [D-365,D), excluding D",
        "same_day_leads_rule": "same-origin full delivery-day forecast profile only; never next day",
        "stress_upper_clipped": False, "observed_prices_used": False,
        "neighbor_observed_prices_used": False, "imputation_performed": False,
        "pit_publication_evidence_verified": False,
        "publication_contract": "caller supplies both zones forecasts available by origin cutoff",
        "normalizations": normalizations,
    }


def mixture_quantiles(
    normal_knots: np.ndarray, spike_knots: np.ndarray, spike_probability: np.ndarray,
    *, knot_probabilities: Sequence[float] = EXPERT_PROBABILITIES,
    output_probabilities: Sequence[float] = OUTPUT_PROBABILITIES,
) -> np.ndarray:
    """Invert (1-p) F_normal + p F_spike; knots define monotone quantile curves.

    Linear interpolation between conditional quantile knots defines each CDF.
    Repeated knots are atoms, not deleted samples. Endpoints at alpha 0 and 1
    define finite empirical/model envelopes. Bisection returns the generalized
    inverse, including atoms; this is never an average of conditional quantiles.
    """
    normal = np.asarray(normal_knots, dtype=float)
    spike = np.asarray(spike_knots, dtype=float)
    probability = np.asarray(spike_probability, dtype=float)
    alpha = np.asarray(knot_probabilities, dtype=float)
    output = np.asarray(output_probabilities, dtype=float)
    if (normal.ndim != 2 or spike.shape != normal.shape
            or probability.shape != (len(normal),) or alpha.shape != (normal.shape[1],)
            or len(alpha) < 2 or alpha[0] != 0 or alpha[-1] != 1
            or not (np.diff(alpha) > 0).all() or output.ndim != 1
            or not ((output > 0) & (output < 1)).all()):
        raise ValueError("Invalid mixture array shapes or quantile probabilities")
    if (not all(np.isfinite(a).all() for a in (normal, spike, probability, alpha, output))
            or (np.diff(normal, axis=1) < 0).any() or (np.diff(spike, axis=1) < 0).any()
            or ((probability < 0) | (probability > 1)).any()):
        raise ValueError("Mixture requires finite ordered knots and probabilities in [0,1]")
    result = np.empty((len(normal), len(output)), dtype=float)
    for row, weight in enumerate(probability):
        if weight == 0:
            result[row] = np.interp(output, alpha, normal[row])
            continue
        if weight == 1:
            result[row] = np.interp(output, alpha, spike[row])
            continue
        lower = min(normal[row, 0], spike[row, 0])
        upper = max(normal[row, -1], spike[row, -1])
        for column, target in enumerate(output):
            lo, hi = lower, upper
            for _ in range(64):
                middle = (lo + hi) / 2
                cdf = ((1 - weight) * np.interp(middle, normal[row], alpha, left=0, right=1)
                       + weight * np.interp(middle, spike[row], alpha, left=0, right=1))
                if cdf >= target:
                    hi = middle
                else:
                    lo = middle
            result[row, column] = hi
    return result


def _expert(
    train: np.ndarray, residual: np.ndarray, test: np.ndarray, regime: str,
    parameters: dict,
) -> tuple[np.ndarray, dict]:
    count = len(residual)
    audit = {"rows": count, "minimum_model_rows": MINIMUM_EXPERT_ROWS[regime]}
    varying = len(train) > 1 and np.any(np.ptp(train, axis=0) > 0)
    if count >= MINIMUM_EXPERT_ROWS[regime] and varying and np.ptp(residual) > 0:
        from catboost import CatBoostRegressor
        alpha = EXPERT_PROBABILITIES[1:-1]
        model = CatBoostRegressor(
            loss_function="MultiQuantile:alpha=" + ",".join(str(x) for x in alpha),
            **parameters,
        )
        model.fit(train, residual)
        raw = np.asarray(model.predict(test), dtype=float)
        crossing = np.any(np.diff(raw, axis=1) < 0, axis=1)
        internal = np.sort(raw, axis=1)
        audit.update(method="catboost_multiquantile", quantile_crossing_rows=int(crossing.sum()),
                     monotonicity_repair="increasing rearrangement of predicted quantiles")
        # Projection onto the known regime support is not a correction cap:
        # the mixture as a whole has neither a +/-40 nor any other price bound.
        if regime == "normal":
            internal = np.minimum(internal, SPIKE_THRESHOLD)
        else:
            internal = np.maximum(internal, np.nextafter(SPIKE_THRESHOLD, np.inf))
        lower = np.minimum(float(residual.min()), internal[:, 0])
        upper = np.maximum(float(residual.max()), internal[:, -1])
        knots = np.column_stack([lower, internal, upper])
    elif count:
        knots = np.repeat(np.quantile(residual, EXPERT_PROBABILITIES)[None, :], len(test), axis=0)
        audit.update(method="empirical_quantiles",
                     fallback_reason="sparse_regime_or_constant_features_or_targets")
    else:
        # This component has zero empirical gate probability and contributes no
        # mass. The placeholder only keeps a valid mathematical representation.
        point = SPIKE_THRESHOLD + 1 if regime == "spike" else 0.0
        knots = np.full((len(test), len(EXPERT_PROBABILITIES)), point)
        audit.update(method="empty_regime_placeholder", fallback_reason="no_observed_regime_rows")
    audit.update(conditional_support="residual>50" if regime == "spike" else "residual<=50",
                 residual_min=float(residual.min()) if count else None,
                 residual_max=float(residual.max()) if count else None)
    return knots, audit


def fit_predict(
    train_X: pd.DataFrame, train_residual: Sequence[float], test_X: pd.DataFrame,
    train_days: Sequence, *, origin_day: str | date, threads: int = 1,
    iterations: int = 120, seed: int = 20260923,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit one pooled DE/NL origin and return residual Q10/Q50/Q90, gate p, audit.

    Training days must contain >=90 distinct civil days, all in [origin-365,
    origin). They are rejected, not silently trimmed, if this contract fails.
    The caller must pass availability-safe, frozen baseline residuals and use
    chronological weekly refits. This function cannot certify their vintage.
    All features are numeric and finite, with an identical train/test schema.
    No heldout event labels, test prices, class weights or label-dependent
    feature transformations enter predictions.
    """
    origin = pd.Timestamp(origin_day).date()
    if not 1 <= threads <= 2 or not 1 <= iterations <= 120:
        raise ValueError("Resource contract: 1-2 threads and 1-120 trees")
    if (not isinstance(train_X, pd.DataFrame) or not isinstance(test_X, pd.DataFrame)
            or train_X.empty or test_X.empty or not train_X.columns.is_unique
            or not train_X.columns.equals(test_X.columns)):
        raise ValueError("Nonempty matching unique train/test numeric feature schemas required")
    train = train_X.to_numpy(dtype=float)
    test = test_X.to_numpy(dtype=float)
    residual = np.asarray(train_residual, dtype=float)
    parsed_days = pd.DatetimeIndex(pd.to_datetime(list(train_days)))
    if (residual.shape != (len(train),) or len(parsed_days) != len(train)
            or parsed_days.hasnans):
        raise ValueError("Training labels and civil days must match feature rows")
    if not all(np.isfinite(a).all() for a in (train, test, residual)):
        raise ValueError("All features and residuals must be finite; exclude warmup explicitly")
    days = np.asarray(parsed_days.date)
    if np.any(days >= origin) or np.any(days < origin - timedelta(days=365)):
        raise ValueError("Training civil days must lie in [origin-365,origin); future/origin labels forbidden")
    unique_days = sorted(set(days))
    if len(unique_days) < TRAINING_WARMUP_DAYS:
        raise ValueError("At least 90 distinct earlier training days are required")
    target = (residual > SPIKE_THRESHOLD).astype(int)
    positives = int(target.sum())
    negatives = len(target) - positives
    parameters = {
        "iterations": int(iterations), "depth": 4, "learning_rate": .05,
        "l2_leaf_reg": 3.0, "random_seed": int(seed), "thread_count": int(threads),
        "allow_writing_files": False, "verbose": False, "task_type": "CPU",
    }
    gate_audit = {
        "training_spike_rows": positives, "training_normal_rows": negatives,
        "class_weights_used": False, "threshold_eur_mwh": SPIKE_THRESHOLD,
        "calibration_window_days": CALIBRATION_DAYS,
    }
    varying = np.any(np.ptp(train, axis=0) > 0)
    if positives < MINIMUM_GATE_SPIKES or negatives < MINIMUM_GATE_NORMAL or not varying:
        probability = np.full(len(test), positives / len(target), dtype=float)
        gate_audit.update(method="empirical_frequency", calibration_performed=False,
                          fallback_reason="sparse_class_or_constant_features")
    else:
        from catboost import CatBoostClassifier
        calibration = days >= origin - timedelta(days=CALIBRATION_DAYS)
        earlier = ~calibration
        enough = (int(target[earlier].sum()) >= MINIMUM_GATE_SPIKES
                  and int((1 - target[earlier]).sum()) >= MINIMUM_GATE_NORMAL
                  and int(target[calibration].sum()) >= MINIMUM_GATE_SPIKES
                  and int((1 - target[calibration]).sum()) >= MINIMUM_GATE_NORMAL
                  and np.any(np.ptp(train[earlier], axis=0) > 0))
        gate = CatBoostClassifier(loss_function="Logloss", **parameters)
        if enough:
            from sklearn.linear_model import LogisticRegression
            gate.fit(train[earlier], target[earlier])
            calibration_logits = np.asarray(
                gate.predict(train[calibration], prediction_type="RawFormulaVal"), dtype=float,
            ).reshape(-1, 1)
            calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                            random_state=int(seed))
            calibrator.fit(calibration_logits, target[calibration])
            test_logits = np.asarray(gate.predict(test, prediction_type="RawFormulaVal"),
                                     dtype=float).reshape(-1, 1)
            probability = calibrator.predict_proba(test_logits)[:, 1]
            gate_audit.update(method="catboost_platt_calibrated", calibration_performed=True,
                              gate_fit_last_day=max(days[earlier]).isoformat(),
                              calibration_first_day=min(days[calibration]).isoformat(),
                              calibration_last_day=max(days[calibration]).isoformat(),
                              calibration_rows=int(calibration.sum()),
                              calibration_spike_rows=int(target[calibration].sum()),
                              calibration_slope=float(calibrator.coef_[0, 0]),
                              calibration_intercept=float(calibrator.intercept_[0]))
        else:
            gate.fit(train, target)
            probability = gate.predict_proba(test)[:, 1]
            gate_audit.update(method="catboost_uncalibrated", calibration_performed=False,
                              calibration_skip_reason="insufficient_classes_in_chronological_holdout",
                              gate_fit_last_day=max(days).isoformat())
    normal_knots, normal_audit = _expert(train[target == 0], residual[target == 0],
                                         test, "normal", parameters)
    spike_knots, spike_audit = _expert(train[target == 1], residual[target == 1],
                                      test, "spike", parameters)
    quantiles = mixture_quantiles(normal_knots, spike_knots, probability)
    if not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < 0).any():
        raise ValueError("Invalid mixture output distribution")
    audit = {
        "protocol_version": PROTOCOL_VERSION, "origin_day": origin.isoformat(),
        "train_first_day": min(days).isoformat(), "train_last_day": max(days).isoformat(),
        "train_unique_days": len(unique_days), "train_rows": len(train), "test_rows": len(test),
        "feature_names": list(train_X.columns), "model_parameters": parameters,
        "gate": gate_audit, "experts": {"normal": normal_audit, "spike": spike_audit},
        "quantile_probabilities": OUTPUT_PROBABILITIES.tolist(),
        "expert_knot_probabilities": EXPERT_PROBABILITIES.tolist(),
        "combination": "generalized inverse of (1-p)*F_normal+p*F_spike; not average quantiles",
        "output_definition": "residual quantiles; add each to same frozen interaction40 P50",
        "correction_clip": None, "prices_used_as_features": "only caller frozen baseline_p50 if supplied",
        "pit_publication_evidence_verified": False, "prospective_calibration_verified": False,
        "limitations": [
            "Input timestamps alone do not establish forecast publication or observed-label availability.",
            "Two-zone pooling assumes transferable scarcity relationships; country flag retains identity.",
            "Sparse-regime empirical fallback and tree experts cannot reliably extrapolate unseen extremes.",
            "Finite conditional quantile grid and empirical/model endpoint envelopes approximate tails.",
            "Platt holdout calibration can be skipped for sparse classes; no calibration guarantee follows.",
            "No baseline-uncertainty convolution is performed; this replaces the residual distribution.",
            "22 September was inspected to formulate this hypothesis and is not an independent validation.",
        ],
    }
    return quantiles, np.asarray(probability, dtype=float), audit


__all__ = ["PROTOCOL_VERSION", "SPIKE_THRESHOLD", "EXPERT_PROBABILITIES",
           "OUTPUT_PROBABILITIES", "build_features", "mixture_quantiles", "fit_predict"]
