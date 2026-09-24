"""Experimental, cadence-aware port of the production NYX residual recipe.

All learning is daily and uses complete earlier civil days. Feature construction
is deterministic within a delivery day and never receives observations. This
module deliberately does not alter the hourly production validators or caches.
"""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chronos2_hourly.models.residual_corrector import (
    ResidualCorrector, ResidualMetaFeatureBuilder, _matches,
)


QUANTILES = ("q10", "q50", "q90")


def _cadence(frequency: str) -> tuple[str, int]:
    if frequency not in {"h", "15min"}:
        raise ValueError("frequency must be h or 15min.")
    return frequency, 4 if frequency == "15min" else 1


def _complete_index(index: pd.Index, frequency: str, timezone: str) -> pd.DatetimeIndex:
    _cadence(frequency)
    if not isinstance(index, pd.DatetimeIndex) or index.empty or index.tz is None:
        raise ValueError("A nonempty timezone-aware UTC index is required.")
    if str(index.tz) != "UTC" or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Index must be unique, sorted and explicitly UTC.")
    local = index.tz_convert(timezone)
    first, last = local[0].date(), local[-1].date()
    expected = pd.date_range(pd.Timestamp(first, tz=timezone),
                             pd.Timestamp(last+timedelta(days=1), tz=timezone),
                             freq=frequency, inclusive="left").tz_convert("UTC")
    if not index.equals(expected):
        raise ValueError("Only contiguous complete civil days are admitted; no filling or skipped quarters.")
    return index


def _finite_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.columns.has_duplicates:
        raise ValueError(f"{name}: nonempty DataFrame with unique columns required.")
    if not all(isinstance(column, str) for column in frame):
        raise ValueError(f"{name}: string column names required.")
    if any(not (pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])) for c in frame):
        raise ValueError(f"{name}: numeric columns required.")
    result = frame.astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{name}: all values must be finite; no imputation is applied.")
    return result


def _raw(frame: pd.DataFrame, index: pd.DatetimeIndex, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not frame.index.equals(index) or set(frame.columns) != set(QUANTILES):
        raise ValueError(f"{name}: exactly aligned q10/q50/q90 required.")
    result = _finite_frame(frame.loc[:, list(QUANTILES)], name)
    if not ((result.q10 <= result.q50) & (result.q50 <= result.q90)).all():
        raise ValueError(f"{name}: crossed quantiles are forbidden.")
    return result


class CadenceResidualMetaFeatureBuilder(ResidualMetaFeatureBuilder):
    """Retain production feature families with explicit physical-time ramps."""

    def __init__(self, *, frequency: str = "15min", **options: Any):
        self.frequency, self.points_per_hour = _cadence(frequency)
        super().__init__(**options)

    def _calendar(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        hourly = index.floor("h")
        result = super()._calendar(hourly.unique()).reindex(hourly)
        result.index = index
        local = index.tz_convert(self.timezone)
        hour = local.hour.to_numpy(float) + local.minute.to_numpy(float)/60
        result["calendar_local_hour"] = hour
        result["calendar_hour_sin"] = np.sin(2*np.pi*hour/24)
        result["calendar_hour_cos"] = np.cos(2*np.pi*hour/24)
        return result

    def _rich_calendar(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        # Holidays, DST and hourly peak windows have no invented intrahour detail.
        hourly = index.floor("h")
        result = super()._rich_calendar(hourly.unique()).reindex(hourly)
        result.index = index
        return result

    def _daily_profiles(self, frame: pd.DataFrame, *, source_columns) -> pd.DataFrame:
        if self.points_per_hour == 1:
            return super()._daily_profiles(frame, source_columns=source_columns)
        days = pd.Series(frame.index.tz_convert(self.timezone).date, index=frame.index)
        derived = {}
        for column in source_columns:
            values = frame[column].astype(float)
            grouped = values.groupby(days, sort=False)
            mean, low, high = grouped.transform("mean"), grouped.transform("min"), grouped.transform("max")
            position = grouped.cumcount()
            one = grouped.diff(4).mask(position.lt(4), 0.0)
            two = grouped.diff(8).mask(position.lt(8), 0.0)
            features = {
                "day_mean": mean, "day_std": grouped.transform("std", ddof=0),
                "day_min": low, "day_max": high, "day_range": high-low,
                "day_centered": values-mean, "ramp_1h": one, "abs_ramp_1h": one.abs(),
                "ramp_2h": two, "abs_ramp_2h": two.abs(),
                "day_ramp_up_max": one.clip(lower=0).groupby(days, sort=False).transform("max"),
                "day_ramp_down_max": (-one).clip(lower=0).groupby(days, sort=False).transform("max"),
                "day_ramp_abs_max": one.abs().groupby(days, sort=False).transform("max"),
            }
            derived.update({f"{column}__{name}": value for name, value in features.items()})
        return pd.DataFrame(derived, index=frame.index)

    def _build(self, X: pd.DataFrame, expert_predictions: pd.DataFrame):
        """Mirror production assembly, replacing only its hourly boundaries."""
        index = _complete_index(X.index, self.frequency, self.timezone)
        X = _finite_frame(X, "X")
        if not expert_predictions.index.equals(index):
            raise ValueError("Expert forecasts must exactly match X.")
        experts = _finite_frame(expert_predictions, "experts")
        expected = {f"{prefix}__{q}" for prefix in ("base", "chronos2") for q in QUANTILES}
        if set(experts) != expected:
            raise ValueError("This NYX port only admits the base and Chronos q10/q50/q90.")
        selected = X.loc[:, [column for column in X if not self._is_excluded(column)]]
        if selected.empty or set(selected).intersection(experts):
            raise ValueError("Empty or colliding residual feature schema.")
        parts = [selected, experts]
        fundamental_profiles = ()
        if self.include_fundamental_interactions:
            fundamentals, fundamental_profiles = self._fundamental_interactions(selected)
            if not fundamentals.empty:
                parts.append(fundamentals)
        for family in (self._missing_indicators(selected), self._residual_load_interactions(selected),
                       self._expert_disagreement(experts)):
            if not family.empty:
                parts.append(family)
        spreads = self._chronos_spreads(experts)
        if not spreads.empty:
            parts.append(spreads)
        for enabled, builder in ((self.include_calendar, self._calendar),
                                 (self.include_rich_calendar, self._rich_calendar)):
            if enabled:
                family = builder(index)
                family = family.loc[:, [c for c in family if c not in X and c not in experts]]
                if not family.empty:
                    parts.append(family)
        base = pd.concat(parts, axis=1)
        if self.include_daily_profiles:
            if self.profile_columns is not None:
                columns = list(self.profile_columns)
            else:
                columns = [c for c in selected if _matches(c, self._residual_patterns)]
                columns += [c for c in experts if c.rsplit("__", 1)[0].lower().startswith(self.chronos_prefixes)]
                columns += [c for c in spreads if c.endswith("__interval_width")]
                columns += list(fundamental_profiles)
            if set(columns)-set(base):
                raise ValueError("Missing declared daily-profile columns.")
            profiles = self._daily_profiles(base, source_columns=columns)
            if not profiles.empty:
                parts.append(profiles)
        result = _finite_frame(pd.concat(parts, axis=1), "meta features")
        return result, tuple(column for column in X if column not in selected)


def build_meta_features(
    X: pd.DataFrame, raw: pd.DataFrame, *, frequency: str = "15min",
    timezone: str = "Europe/Paris", primary_country: str = "FR",
    feature_builder_options: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build no-learn, complete-day features; neither labels nor fitted statistics."""
    index = _complete_index(X.index, frequency, timezone)
    base = _raw(raw, index, "raw")
    options = dict(feature_builder_options or {})
    options.update(timezone=timezone, rich_calendar_primary_country=primary_country)
    options.setdefault("include_rich_calendar", True)
    options.setdefault("include_daily_profiles", True)
    if options.get("exclude_historical_prices", True) is not True or options.get("exclude_day_of_year", True) is not True:
        raise ValueError("Historical-price and day-of-year exclusions are required.")
    if any(c.casefold() in {"actual", "target", "observed", "price", "nyx_q50"} for c in X):
        raise ValueError("Observed prices or another final model cannot enter residual features.")
    builder = CadenceResidualMetaFeatureBuilder(frequency=frequency, **options)
    experts = pd.concat([base.add_prefix("base__"), base.add_prefix("chronos2__")], axis=1)
    return builder._build(X, experts)[0]


def _frame_hash(frame: pd.DataFrame | pd.Series) -> str:
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    if isinstance(frame, pd.DataFrame):
        digest.update(json.dumps(list(frame.columns)).encode())
    return digest.hexdigest()


def fit_predict_day(
    meta_train: pd.DataFrame, actual_train: pd.Series, raw_train: pd.DataFrame,
    meta_day: pd.DataFrame, raw_day: pd.DataFrame, *, recipe: Mapping[str, Any],
    frequency: str = "15min", timezone: str = "Europe/Paris",
    max_lookback_days: int = 365, minimum_training_days: int = 30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit one fresh production CatBoost and apply its common capped shift.

    Callers supply their chosen trailing training window. No implicit clipping,
    fitting on D, model reuse, observation fallback or alternate backend occurs.
    Below the declared cold-start threshold, output is explicitly the raw base.
    """
    _, points = _cadence(frequency)
    if type(max_lookback_days) is not int or max_lookback_days < 1 or type(minimum_training_days) is not int or minimum_training_days < 1:
        raise ValueError("Training-day bounds must be positive integers.")
    train_index = _complete_index(meta_train.index, frequency, timezone)
    day_index = _complete_index(meta_day.index, frequency, timezone)
    train = _finite_frame(meta_train, "meta_train")
    predict = _finite_frame(meta_day, "meta_day")
    if list(train) != list(predict):
        raise ValueError("Prediction feature schema must equal the training schema in order.")
    raw_train = _raw(raw_train, train_index, "raw_train")
    raw_day = _raw(raw_day, day_index, "raw_day")
    days = pd.Index(train_index.tz_convert(timezone).date).unique()
    target_days = pd.Index(day_index.tz_convert(timezone).date).unique()
    if len(target_days) != 1:
        raise ValueError("Exactly one complete target day is required.")
    target_day = target_days[0]
    if days[-1] != target_day-timedelta(days=1) or days[0] < target_day-timedelta(days=max_lookback_days):
        raise ValueError("Training must end D-1 and remain entirely inside the trailing lookback.")
    if not isinstance(actual_train, pd.Series) or not actual_train.index.equals(train_index):
        raise ValueError("Only exactly aligned training observations are admitted.")
    actual = pd.to_numeric(actual_train, errors="raise").astype(float)
    if not np.isfinite(actual.to_numpy()).all():
        raise ValueError("Training observations must all be finite.")
    options = dict(recipe)
    if options.pop("enabled", True) is not True or options.pop("base_model", "chronos2") != "chronos2":
        raise ValueError("An enabled Chronos residual recipe is required.")
    options.pop("feature_builder", None)  # Already applied by build_meta_features.
    if options.get("backend") != "catboost":
        raise ValueError("This experiment requires the genuine CatBoost backend.")
    options["min_training_rows"] = int(options.get("min_training_rows", 720))*points
    corrector = ResidualCorrector(**options)
    audit = {
        "delivery_day": str(target_day), "frequency": frequency, "timezone": timezone,
        "fit_start_day": str(days[0]), "fit_end_day": str(days[-1]),
        "training_rows": len(train), "training_days": len(days),
        "minimum_training_rows": corrector.min_training_rows,
        "minimum_training_days": minimum_training_days, "max_lookback_days": max_lookback_days,
        "feature_count": len(train.columns), "feature_columns": list(train.columns),
        "training_features_sha256": _frame_hash(train), "training_actual_sha256": _frame_hash(actual),
        "training_raw_sha256": _frame_hash(raw_train), "forecast_features_sha256": _frame_hash(predict),
        "forecast_raw_sha256": _frame_hash(raw_day), "recipe": dict(recipe),
        "label": "actual_minus_chronos_q50", "refit_cadence_days": 1,
        "quantile_policy": "same_additive_shift_q10_q50_q90",
        "target_observations_used": False, "causality_violations": 0,
    }
    if len(train) < corrector.min_training_rows or len(days) < minimum_training_days:
        audit.update(generation_source="identity_chronos_cold_start", backend=None,
                     clipped_count=0, correction_abs_max=0.0)
        return raw_day.copy(), audit
    corrector.backend_ = corrector._resolve_backend()
    model = corrector._new_model()
    model.fit(train, (actual-raw_train.q50).to_numpy(float))
    raw_correction = np.asarray(model.predict(predict), dtype=float)
    if raw_correction.shape != (len(predict),) or not np.isfinite(raw_correction).all():
        raise ValueError("CatBoost returned invalid corrections.")
    scaled = raw_correction*corrector.correction_scale
    bounds = corrector.correction_bounds_
    applied = np.clip(scaled, *bounds) if bounds is not None else scaled
    result = raw_day.add(applied, axis=0)
    audit.update(generation_source="daily_prequential_refit", backend="catboost",
                 clipped_count=int(np.count_nonzero(applied != scaled)),
                 correction_abs_max=float(np.max(np.abs(applied))),
                 correction_mean=float(np.mean(applied)),
                 effective_backend_parameters=model.get_params() if hasattr(model, "get_params") else {})
    result.attrs["residual_correction"] = pd.Series(applied, index=day_index)
    return result, audit


__all__ = ["CadenceResidualMetaFeatureBuilder", "build_meta_features", "fit_predict_day"]
