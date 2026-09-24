"""Isolated native-cadence replay of the production daily Kalman policy.

Forecasts for D use only complete prior delivery days. Q and persistence are
daily, unchanged between hourly and quarter-hourly execution. Observations are
assimilated one native step at a time only after that day's forecast is frozen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from chronos2_hourly.kalman_covariates import KalmanCovariateConfig, materialize_kalman_covariates
from chronos2_hourly.kalman_residual import (
    KalmanResidualConfig, _Candidate, _candidate_market_feature_columns,
    _governance_audit_maps, _governance_choice, _market_features, _observation_variance,
)


QUANTILES = ("q10", "q50", "q90")


@dataclass(frozen=True)
class KalmanDayResult:
    predictions: pd.DataFrame
    audit: dict[str, Any]
    candidate_predictions: pd.DataFrame
    state_audit: pd.DataFrame


def _frame(value: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or value.empty or value.columns.has_duplicates:
        raise ValueError(f"{name}: nonempty frame with unique columns required.")
    frame = value.copy(deep=True)
    if "delivery_start_utc" in frame:
        frame = frame.set_index("delivery_start_utc")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{name}: ordered, unique, timezone-aware timestamps required.")
    frame.index = index.tz_convert("UTC")
    frame.index.name = "delivery_start_utc"
    return frame


def _day_index(day, frequency: str, timezone: str) -> pd.DatetimeIndex:
    first = pd.Timestamp(day).tz_localize(timezone)
    last = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(timezone)
    return pd.date_range(first, last, freq=frequency, inclusive="left").tz_convert("UTC")


def _features(frame: pd.DataFrame, *, upstream_model: str, timezone: str, is_future: bool) -> pd.DataFrame:
    result = frame.copy(deep=True)
    required = [f"{upstream_model}__{q}" for q in QUANTILES]
    if not is_future:
        required.append("actual")
    if any(column not in result for column in required):
        raise ValueError("Upstream quantiles and historical actuals are required.")
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    if not np.isfinite(result[required].to_numpy()).all():
        raise ValueError("Upstream quantiles and historical actuals must all be finite.")
    lower, median, upper = (result[f"{upstream_model}__{q}"] for q in QUANTILES)
    if not ((lower <= median) & (median <= upper)).all():
        raise ValueError("Crossed upstream quantiles are forbidden.")
    if is_future:
        if "actual" in result and result.actual.notna().any():
            raise ValueError("Future actual labels must be absent or entirely NaN.")
        result["actual"] = np.nan
    if "residual_correction" in result:
        shift = pd.to_numeric(result.residual_correction, errors="raise").astype(float)
    elif "chronos2__q50" in result:
        shift = median - pd.to_numeric(result["chronos2__q50"], errors="raise").astype(float)
    else:
        raise ValueError("The residual shift or pre-residual Chronos median is required.")
    if not np.isfinite(shift.to_numpy()).all():
        raise ValueError("The residual shift must be finite.")
    if "residual_correction" in result and "chronos2__q50" in result:
        expected = median - pd.to_numeric(result["chronos2__q50"], errors="raise").astype(float)
        if not np.allclose(shift, expected, rtol=1e-10, atol=1e-10):
            raise ValueError("Residual correction disagrees with its Chronos upstream.")
    local = result.index.tz_convert(timezone)
    hour = local.hour + local.minute / 60.0
    result["_local_day"] = local.date
    result["_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    result["_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    result["_hour_sin2"] = np.sin(4.0 * np.pi * hour / 24.0)
    result["_hour_cos2"] = np.cos(4.0 * np.pi * hour / 24.0)
    result["_base_q50"] = median
    result["_interval_width"] = upper - lower
    result["_residual_shift"] = shift
    result["_pre_residual_q50"] = median - shift
    result["_upstream_error"] = result.actual - median
    return result


def forecast_day(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    frequency: str,
    timezone: str = "Europe/Paris",
    covariate_columns: Sequence[str] = (),
    covariate_config: KalmanCovariateConfig | None = None,
    config: KalmanResidualConfig | None = None,
    lookback_days: int = 365,
    upstream_model: str = "residual_corrected",
    output_model: str = "residual_kalman",
) -> KalmanDayResult:
    """Refit on available complete prior days (at most 365), then forecast D.

    The six original nuclear recipe covariates are expected in both frames.
    Production mean/spread features are materialized from those same inputs.
    A shorter archive is explicitly audited, never described as rolling-365.
    Latest historical labels do not establish historical publication vintages.
    """
    if frequency not in {"h", "15min"}:
        raise ValueError("frequency must be h or 15min.")
    if type(lookback_days) is not int or not 1 <= lookback_days <= 365:
        raise ValueError("lookback_days must be an integer between 1 and 365.")
    policy = config or KalmanResidualConfig()
    policy.validate()
    if covariate_config is None:
        from chronos2_hourly.nuclear_forecast import nuclear_kalman_covariate_config
        covariate_config = nuclear_kalman_covariate_config()
    covariate_config.validate()
    if covariate_columns and tuple(covariate_columns) != tuple(covariate_config.input_columns):
        raise ValueError("covariate_columns must match the configured original input columns in order.")
    if any(item.kind == "ramp" for item in covariate_config.derived):
        raise ValueError("This fixed recipe has no ramps; physical ramp durations require a separate protocol.")
    training = _frame(history, name="history")
    target = _frame(future, name="future")
    target_days = pd.Index(target.index.tz_convert(timezone).date).unique()
    if len(target_days) != 1 or not target.index.equals(_day_index(target_days[0], frequency, timezone)):
        raise ValueError("Future must be exactly one complete native-cadence civil day, including DST.")
    target_day = target_days[0]
    start = target.index[0]
    if training.index.max() >= start:
        raise ValueError("Historical labels must be strictly before delivery D.")
    available_days = pd.Index(training.index.tz_convert(timezone).date).unique()
    lower = max(pd.Timestamp(available_days[0]), pd.Timestamp(target_day) - pd.Timedelta(days=lookback_days))
    training = training.loc[training.index >= lower.tz_localize(timezone).tz_convert("UTC")].copy()
    expected = pd.date_range(lower.tz_localize(timezone), pd.Timestamp(target_day).tz_localize(timezone),
                             freq=frequency, inclusive="left").tz_convert("UTC")
    if not training.index.equals(expected):
        raise ValueError("History must cover complete contiguous civil days through D-1, with no filled gaps.")
    training = _features(training, upstream_model=upstream_model, timezone=timezone, is_future=False)
    target = _features(target, upstream_model=upstream_model, timezone=timezone, is_future=True)
    feature_frame = pd.concat([training, target])
    covariates = materialize_kalman_covariates(feature_frame, covariate_config, timezone=timezone)
    used_covariates = covariate_config.feature_columns
    if not np.isfinite(covariates.loc[:, list(used_covariates)].to_numpy()).all():
        raise ValueError("All historical and future model covariates must be finite; no neutral filling.")
    for column in used_covariates:
        feature_frame[column] = covariates[column]
    training = feature_frame.loc[training.index]
    target = feature_frame.loc[target.index]
    training_days = tuple(pd.Index(training._local_day).unique())
    calibration_days = set(training_days)
    market, scalers = _market_features(feature_frame, covariate_columns=used_covariates,
                                      calibration_days=calibration_days, clip=policy.market_feature_clip)
    variance = _observation_variance(training, calibration_days=calibration_days)
    candidates = {
        kind: _Candidate(kind=kind, config=policy, observation_variance=variance, market=market,
                         market_feature_columns=_candidate_market_feature_columns(
                             kind, covariate_config=covariate_config, covariate_columns=used_covariates))
        for kind in policy.candidate_kinds
    }
    governance_days = set(training_days[-policy.governance_lookback_days:])
    realised_rows = []
    for local_day, block in training.groupby("_local_day", sort=True):
        for candidate in candidates.values():
            candidate.transition_day()
        if local_day in governance_days:
            realised_rows.append(pd.DataFrame({"local_day": local_day, "actual": block.actual.to_numpy(),
                "base": block[f"{upstream_model}__q50"].to_numpy(),
                **{f"raw::{kind}": np.clip(candidate.raw_correction(block), -policy.shift_clip_eur_mwh,
                                           policy.shift_clip_eur_mwh) for kind, candidate in candidates.items()}}))
        for _, row in block.iterrows():
            for candidate in candidates.values():
                candidate.update_hour_rolling(row)
    before = {kind: candidate.state_values() for kind, candidate in candidates.items()}
    for candidate in candidates.values():
        candidate.transition_day()
    frozen = {kind: candidate.state_values() for kind, candidate in candidates.items()}
    raw = {kind: candidate.raw_correction(target) for kind, candidate in candidates.items()}
    clipped = {kind: np.clip(value, -policy.shift_clip_eur_mwh, policy.shift_clip_eur_mwh) for kind, value in raw.items()}
    realised = pd.concat(realised_rows, ignore_index=True)
    selected, weight, losses = _governance_choice(realised, candidate_kinds=tuple(candidates), config=policy)
    candidate_losses, governance = _governance_audit_maps(losses, candidate_kinds=tuple(candidates))
    selected_raw = np.zeros(len(target)) if selected == "identity" else clipped[selected]
    applied = weight * selected_raw
    predictions = pd.DataFrame({f"{output_model}__{q}": target[f"{upstream_model}__{q}"].to_numpy() + applied
                                for q in QUANTILES}, index=target.index)
    predictions["kalman_raw_correction"] = selected_raw
    predictions["kalman_weight"] = weight
    predictions["kalman_correction"] = applied
    predictions["kalman_selected_filter"] = selected
    candidate_predictions = pd.DataFrame({f"{kind}__q50": target[f"{upstream_model}__q50"].to_numpy() + value
                                         for kind, value in clipped.items()}, index=target.index)
    audit = {
        "delivery_day": str(target_day), "frequency": frequency,
        "step_minutes": 15 if frequency == "15min" else 60, "target_points": len(target),
        "training_window_start": str(training_days[0]), "training_window_end": str(training_days[-1]),
        "training_window_days": len(training_days), "training_window_points": len(training),
        "available_history_days": len(available_days), "lookback_cap_days": lookback_days,
        "short_history": len(training_days) < 365, "full_365_day_history": len(training_days) >= 365,
        "history_policy": "all_available_complete_days_capped_at_365_including_declared_residual_cold_start",
        "last_training_timestamp_utc": training.index[-1].isoformat(),
        "target_first_timestamp_utc": target.index[0].isoformat(), "target_observations_assimilated": 0,
        "publication_vintages_verified": False, "candidate_kinds": list(candidates),
        "filter_parameters": asdict(policy), "selected_filter": selected, "selected_weight": float(weight),
        "transition_unit": "civil_day", "daily_transitions_per_candidate": len(training_days)+1,
        "observation_updates_per_candidate": len(training), "q_rescaled_for_frequency": False,
        "observation_variance": float(variance), "calibration_max_day": str(training_days[-1]),
        "market_scalers": scalers, "covariate_config": covariate_config.to_dict(),
        "candidate_feature_names": {kind: list(c.feature_names) for kind, c in candidates.items()},
        "governance_realised_days": int(realised.local_day.nunique()),
        "governance_realised_points": len(realised), "candidate_trailing_mae": candidate_losses,
        "governance_diagnostics": governance, "baseline_trailing_mae": losses.get("identity"),
        "applied_correction_abs_max": float(np.abs(applied).max()),
        "clipped_target_corrections": sum(int((np.abs(value)>policy.shift_clip_eur_mwh).sum()) for value in raw.values()),
        "state_before_transition": before, "state_at_forecast": frozen,
    }
    states = pd.DataFrame([{"filter_kind": kind, "state_before": frozen[kind], "state_after": candidate.state_values(),
                            "innovation_clips_total": candidate.innovation_clips,
                            "minimum_covariance_eigenvalue": candidate.minimum_eigenvalue,
                            "covariance_repairs_total": candidate.covariance_repairs,
                            "target_observations_assimilated": 0} for kind, candidate in candidates.items()])
    return KalmanDayResult(predictions, audit, candidate_predictions, states)
