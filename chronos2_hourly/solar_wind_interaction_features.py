"""One preregistered, causal low-wind/low-solar/high-residual-load feature.

For each civil delivery day D, normalization uses complete hourly forecast
profiles from [D-365, D), never D or a later day. Wind's q75 includes genuine
zero forecasts; solar's q75 uses strictly positive forecasts to avoid a zero
nighttime scale. With w75, s75, r50 and r90 fitted on that history:

    low_wind = clip(1 - wind / w75, 0, 1)
    low_solar = clip(1 - solar / s75, 0, 1)
    high_residual = clip((residual - r50) / (r90 - r50), 0, 1)
    stress = low_wind * low_solar * high_residual

The first 14 complete historical days are a documented zero-score warmup,
not imputed generation. Invalid sources always fail; invalid denominators
fail once warmup is complete. The caller must supply same-vintage PIT
forecasts: validating timestamps here does not certify publication history.
No prices, labels, fitted model, external source or filesystem is accessed.
"""
from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "solar_wind_interaction_features_v1"
TRAINING_WINDOW_DAYS = 365
MINIMUM_HISTORY_DAYS = 14
GENERATION_SCALE_QUANTILE = 0.75
RESIDUAL_LOW_QUANTILE = 0.50
RESIDUAL_HIGH_QUANTILE = 0.90
QUANTILE_METHOD = "linear"
SUPPORTED_ZONES = frozenset({"DE", "NL"})


class SolarWindInteractionFeatureError(ValueError):
    """The isolated feature's causal input or normalization contract failed."""


def build_interaction(
    covariates: pd.DataFrame, zone: str, timezone: str
) -> tuple[pd.DataFrame, dict]:
    """Return one score column and a JSON-serializable normalization audit.

    ``covariates`` requires an ordered, unique UTC DatetimeIndex and complete,
    consecutive local delivery days, including physical 23/25-hour DST days.
    Only ``{zone}_wind_generation_fcst``, ``{zone}_solar_generation_fcst`` and
    ``{zone}_residual_load_fcst`` are read. Other columns are ignored unchanged.
    All three selected columns must contain finite numeric forecasts in GW;
    generation must be nonnegative. Residual load may legitimately be negative.
    Appending or changing later forecast days cannot change any earlier score.
    """
    if not isinstance(zone, str) or zone.upper() not in SUPPORTED_ZONES:
        raise SolarWindInteractionFeatureError("Only DE and NL are supported.")
    zone = zone.upper()
    if not isinstance(timezone, str):
        raise SolarWindInteractionFeatureError("An explicit IANA timezone is required.")
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise SolarWindInteractionFeatureError("Invalid IANA timezone.") from exc
    if not isinstance(covariates, pd.DataFrame) or covariates.empty:
        raise SolarWindInteractionFeatureError("A nonempty covariate DataFrame is required.")
    if not covariates.columns.is_unique:
        raise SolarWindInteractionFeatureError("Duplicate covariate columns are forbidden.")
    index = covariates.index
    if not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC":
        raise SolarWindInteractionFeatureError("The index must be a UTC DatetimeIndex.")
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise SolarWindInteractionFeatureError("UTC timestamps must be finite, unique and ordered.")
    if not index.equals(index.floor("h")):
        raise SolarWindInteractionFeatureError("Physical hourly timestamps must be hour-aligned.")

    columns = [
        f"{zone.lower()}_wind_generation_fcst",
        f"{zone.lower()}_solar_generation_fcst",
        f"{zone.lower()}_residual_load_fcst",
    ]
    missing = sorted(set(columns).difference(covariates.columns))
    if missing:
        raise SolarWindInteractionFeatureError(f"Missing required forecast columns: {missing}.")
    try:
        values = covariates.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise SolarWindInteractionFeatureError("Selected forecasts must be numeric.") from exc
    if not np.isfinite(values).all():
        raise SolarWindInteractionFeatureError("Selected forecasts must be finite; no imputation is allowed.")
    if (values[:, :2] < 0).any():
        raise SolarWindInteractionFeatureError("Wind and solar generation forecasts must be nonnegative.")

    local_days = index.tz_convert(timezone).date
    days = list(dict.fromkeys(local_days))
    start = pd.Timestamp(days[0]).tz_localize(timezone)
    end = pd.Timestamp(days[-1] + timedelta(days=1)).tz_localize(timezone)
    expected = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    if not index.equals(expected):
        raise SolarWindInteractionFeatureError(
            "Complete consecutive local delivery days are required, preserving physical DST hours."
        )
    boundaries = np.r_[0, np.flatnonzero(local_days[1:] != local_days[:-1]) + 1, len(index)]
    score = np.zeros(len(index), dtype=float)
    daily: list[dict] = []
    for day_number, day in enumerate(days):
        history_first = max(0, day_number - TRAINING_WINDOW_DAYS)
        left, right = int(boundaries[history_first]), int(boundaries[day_number])
        history = values[left:right]
        history_days = day_number - history_first
        day_left, day_right = int(boundaries[day_number]), int(boundaries[day_number + 1])
        positive_solar = history[history[:, 1] > 0, 1]
        record = {
            "delivery_day": day.isoformat(),
            "physical_hours": day_right - day_left,
            "normalization_window_start_day": (day - timedelta(days=TRAINING_WINDOW_DAYS)).isoformat(),
            "normalization_window_end_day_exclusive": day.isoformat(),
            "normalization_training_first_day": days[history_first].isoformat() if history_days else None,
            "normalization_training_last_day": days[day_number - 1].isoformat() if history_days else None,
            "normalization_training_complete_days": history_days,
            "normalization_training_hours": right - left,
            "wind_scale_sample_hours": right - left,
            "solar_scale_positive_sample_hours": int(len(positive_solar)),
            "wind_q75_gw": None,
            "solar_positive_q75_gw": None,
            "residual_q50_gw": None,
            "residual_q90_gw": None,
            "residual_span_gw": None,
            "status": "warmup_zero_score",
        }
        if history_days >= MINIMUM_HISTORY_DAYS:
            if not len(positive_solar):
                raise SolarWindInteractionFeatureError(f"{day}: no positive historical solar scale after warmup.")
            wind_scale = float(np.quantile(history[:, 0], GENERATION_SCALE_QUANTILE, method=QUANTILE_METHOD))
            solar_scale = float(np.quantile(positive_solar, GENERATION_SCALE_QUANTILE, method=QUANTILE_METHOD))
            with np.errstate(over="ignore", invalid="ignore"):
                residual_low, residual_high = np.quantile(
                    history[:, 2], [RESIDUAL_LOW_QUANTILE, RESIDUAL_HIGH_QUANTILE], method=QUANTILE_METHOD
                )
                residual_low, residual_high = float(residual_low), float(residual_high)
                residual_span = residual_high - residual_low
            if (not np.isfinite([wind_scale, solar_scale, residual_low, residual_high, residual_span]).all()
                    or wind_scale <= 0 or solar_scale <= 0 or residual_span <= 0):
                raise SolarWindInteractionFeatureError(f"{day}: invalid normalization denominators after warmup.")
            current = values[day_left:day_right]
            # Clipping before subtraction/division is algebraically identical,
            # and avoids overflow for finite out-of-range forecast values.
            low_wind = 1.0 - np.minimum(current[:, 0], wind_scale) / wind_scale
            low_solar = 1.0 - np.minimum(current[:, 1], solar_scale) / solar_scale
            high_residual = (np.clip(current[:, 2], residual_low, residual_high) - residual_low) / residual_span
            score[day_left:day_right] = np.clip(low_wind * low_solar * high_residual, 0.0, 1.0)
            record.update(
                wind_q75_gw=wind_scale, solar_positive_q75_gw=solar_scale,
                residual_q50_gw=residual_low, residual_q90_gw=residual_high,
                residual_span_gw=residual_span, status="active",
            )
        record.update(score_min=float(score[day_left:day_right].min()),
                      score_max=float(score[day_left:day_right].max()),
                      score_positive_hours=int(np.count_nonzero(score[day_left:day_right] > 0)))
        daily.append(record)

    name = f"{zone.lower()}_low_wind_solar_stress"
    result = pd.DataFrame({name: score}, index=index.copy())
    audit = {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION,
        "zone": zone, "timezone": timezone, "rows": len(index), "feature_column": name,
        "source_columns_read": columns, "source_units": "GW",
        "training_window_days": TRAINING_WINDOW_DAYS, "minimum_history_days": MINIMUM_HISTORY_DAYS,
        "generation_scale_quantile": GENERATION_SCALE_QUANTILE,
        "residual_low_quantile": RESIDUAL_LOW_QUANTILE, "residual_high_quantile": RESIDUAL_HIGH_QUANTILE,
        "quantile_method": QUANTILE_METHOD,
        "formula": "clip(1-W/q75_W,0,1)*clip(1-S/q75_positive_S,0,1)*clip((RL-q50_RL)/(q90_RL-q50_RL),0,1)",
        "wind_scale_sample": "all historical forecast hours, including genuine zeros",
        "solar_scale_sample": "strictly positive historical solar forecasts only",
        "normalization_causality": "complete civil delivery days in [D-365,D); D excluded",
        "forecast_source_cutoff_contract": "caller must provide forecasts issued by civil D-1 08:00; not verified here",
        "pit_publication_evidence_verified": False,
        "source_reads_performed": False, "model_fit_performed": False,
        "prices_or_targets_used": False, "imputation_performed": False,
        "warmup_days": sum(record["status"] == "warmup_zero_score" for record in daily),
        "active_days": sum(record["status"] == "active" for record in daily),
        "normalizations": daily,
    }
    return result, audit


__all__ = ["PROTOCOL_VERSION", "TRAINING_WINDOW_DAYS", "MINIMUM_HISTORY_DAYS",
           "GENERATION_SCALE_QUANTILE", "RESIDUAL_LOW_QUANTILE", "RESIDUAL_HIGH_QUANTILE",
           "QUANTILE_METHOD", "SolarWindInteractionFeatureError", "build_interaction"]
