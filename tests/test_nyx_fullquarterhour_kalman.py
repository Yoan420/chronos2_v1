"""Synthetic-only tests of the isolated full-chain native Kalman adapter."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import kalman_residual as production
from chronos2_hourly.nuclear_forecast import nuclear_kalman_covariate_config
from nyx_fullquarterhour.kalman import _features, forecast_day


def fixture_frames(day="2026-06-18", days=4, frequency="15min"):
    start = (pd.Timestamp(day)-pd.Timedelta(days=days)).tz_localize("Europe/Paris")
    split = pd.Timestamp(day).tz_localize("Europe/Paris")
    end = (pd.Timestamp(day)+pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    index = pd.date_range(start, end, freq=frequency, inclusive="left").tz_convert("UTC")
    t = np.arange(len(index), dtype=float)/(4 if frequency == "15min" else 1)
    base = 40 + 5*np.sin(t/24)
    shift = 1.5+np.cos(t/8)
    values = pd.DataFrame({"actual": base+shift+2+np.sin(t/12), "chronos2__q50": base,
                           "residual_correction": shift,
                           "residual_corrected__q10": base+shift-8,
                           "residual_corrected__q50": base+shift,
                           "residual_corrected__q90": base+shift+12}, index=index)
    values.index.name = "delivery_start_utc"
    for n, column in enumerate(nuclear_kalman_covariate_config().input_columns):
        values[column] = 10+n+2*np.sin(t/(n+3))
    history = values.loc[values.index < split.tz_convert("UTC")].copy()
    future = values.loc[values.index >= split.tz_convert("UTC")].drop(columns="actual").copy()
    return history, future


def test_hourly_adapter_matches_production_rolling_fit_with_all_five_candidates():
    history, future = fixture_frames(days=4, frequency="h")
    config = replace(production.KalmanResidualConfig(), governance_minimum_days=2)
    covariate_config = nuclear_kalman_covariate_config()
    result = forecast_day(history, future, frequency="h", config=config)
    cols = list(covariate_config.input_columns)
    covariates = pd.concat([history[cols], future[cols]]).rename_axis("timestamp").reset_index()
    training, used, _ = production._normalise_input(
        history.drop(columns=cols).reset_index(), upstream_model="residual_corrected", timezone="Europe/Paris",
        covariates=covariates, covariate_config=covariate_config)
    target = production._normalise_future_input(
        future.drop(columns=cols).reset_index(), upstream_model="residual_corrected", timezone="Europe/Paris",
        covariates=covariates, covariate_columns=used, covariate_config=covariate_config)
    expected = production._fit_rolling_target_day(
        training_frame=training, training_days=list(pd.Index(training._local_day).unique()),
        target_block=target, target_day=target._local_day.iloc[0], timezone="Europe/Paris",
        upstream_model="residual_corrected", output_model="residual_kalman", config=config,
        covariate_columns=used,
        candidate_feature_columns={kind: production._candidate_market_feature_columns(
            kind, covariate_config=covariate_config, covariate_columns=used) for kind in config.candidate_kinds})
    pd.testing.assert_frame_equal(result.predictions, expected["predictions"], check_exact=True, check_freq=False)
    pd.testing.assert_frame_equal(result.candidate_predictions, expected["candidate_predictions"], check_exact=True, check_freq=False)
    assert result.audit["observation_variance"] == expected["observation_variance"]
    assert result.audit["market_scalers"] == expected["market_scalers"]
    assert result.audit["state_at_forecast"] == expected["state_after_transition"]
    assert len(result.audit["candidate_kinds"]) == 5
    features = result.audit["candidate_feature_names"]["linear_market"]
    assert "covariate::residual_load_mean" in features
    assert "covariate::residual_load_spread" in features
    assert "covariate::fr_nuclear_generation_fcst_gw" in features


@pytest.mark.parametrize("day,points", [("2026-03-29", 92), ("2026-06-18", 96), ("2026-10-25", 100)])
def test_native_day_grid_daily_transition_and_quantile_width_preserved(day, points):
    history, future = fixture_frames(day, days=2)
    result = forecast_day(history, future, frequency="15min")
    assert len(result.predictions) == points
    assert result.predictions.index.equals(future.index)
    assert result.audit["training_window_days"] == 2
    assert result.audit["daily_transitions_per_candidate"] == 3
    assert result.audit["observation_updates_per_candidate"] == len(history)
    assert result.audit["q_rescaled_for_frequency"] is False
    assert result.audit["filter_parameters"]["q_over_r"] == .001
    assert result.audit["filter_parameters"]["ukf_scale_persistence"] == .99
    assert result.audit["target_observations_assimilated"] == 0
    assert result.audit["selected_filter"] == "identity"  # fewer than 14 realised days
    assert result.audit["short_history"] is True
    for row in result.state_audit.itertuples():
        assert row.state_before == row.state_after
    for q in ("q10", "q50", "q90"):
        np.testing.assert_allclose(result.predictions[f"residual_kalman__{q}"]-
                                   future[f"residual_corrected__{q}"], result.predictions.kalman_correction)
    prepared = _features(future, upstream_model="residual_corrected", timezone="Europe/Paris", is_future=True)
    assert prepared.actual.isna().all()
    assert prepared._upstream_error.isna().all()
    assert prepared._hour_sin.iloc[1] != prepared._hour_sin.iloc[0]


def test_future_feature_changes_cannot_change_calibration_or_prior_state_and_inputs_unchanged():
    history, future = fixture_frames(days=3)
    history_before, future_before = history.copy(deep=True), future.copy(deep=True)
    first = forecast_day(history, future, frequency="15min")
    changed = future.copy()
    for column in nuclear_kalman_covariate_config().input_columns:
        changed[column] += 10000
    for q in ("q10", "q50", "q90"):
        changed[f"residual_corrected__{q}"] += 1000
    changed["chronos2__q50"] += 1000
    second = forecast_day(history, changed, frequency="15min")
    assert first.audit["market_scalers"] == second.audit["market_scalers"]
    assert first.audit["observation_variance"] == second.audit["observation_variance"]
    assert first.audit["state_at_forecast"] == second.audit["state_at_forecast"]
    assert first.audit["calibration_max_day"] == "2026-06-17"
    pd.testing.assert_frame_equal(history, history_before)
    pd.testing.assert_frame_equal(future, future_before)


@pytest.mark.parametrize("defect", ["future_actual", "history_delivery_label", "missing_quarter", "duplicate",
                                   "missing_history_day", "nan_covariate", "nan_actual", "crossed_quantiles"])
def test_incomplete_or_leaking_inputs_fail_closed(defect):
    history, future = fixture_frames(days=3)
    if defect == "future_actual":
        future["actual"] = 10000
    elif defect == "history_delivery_label":
        history = pd.concat([history, future.iloc[:1].assign(actual=10000)])
    elif defect == "missing_quarter":
        future = future.drop(future.index[2])
    elif defect == "duplicate":
        future = pd.concat([future.iloc[:1], future])
    elif defect == "missing_history_day":
        history = history.drop(history.index[96:192])
    elif defect == "nan_covariate":
        future.iloc[0, future.columns.get_loc("fr_residual_load_fcst")] = np.nan
    elif defect == "nan_actual":
        history.iloc[0, history.columns.get_loc("actual")] = np.nan
    else:
        future["residual_corrected__q10"] = future.residual_corrected__q90+1
    with pytest.raises(ValueError):
        forecast_day(history, future, frequency="15min")


def test_rolling_cap_excludes_older_labels_from_fit_and_reports_real_day_count():
    history, future = fixture_frames(days=367, frequency="h")
    policy = replace(production.KalmanResidualConfig(), candidate_kinds=("linear_bias",))
    result = forecast_day(history, future, frequency="h", config=policy)
    changed = history.copy()
    changed.loc[changed.index < pd.Timestamp("2025-06-18", tz="Europe/Paris"), "actual"] += 100000
    again = forecast_day(changed, future, frequency="h", config=policy)
    pd.testing.assert_frame_equal(result.predictions, again.predictions, check_exact=True)
    assert result.audit["training_window_days"] == 365
    assert result.audit["available_history_days"] == 367
    assert result.audit["training_window_start"] == "2025-06-18"
    assert result.audit["full_365_day_history"] is True
    assert result.audit["daily_transitions_per_candidate"] == 366


def test_governed_shift_cap_is_native_price_not_divided_by_four():
    history, future = fixture_frames(days=20)
    history["actual"] += 1000
    config = replace(production.KalmanResidualConfig(), shift_clip_eur_mwh=2,
                     candidate_kinds=("linear_bias",), minimum_gain_eur_mwh=.0001, minimum_relative_gain=0)
    result = forecast_day(history, future, frequency="15min", config=config)
    assert result.audit["selected_filter"] == "linear_bias"
    assert result.predictions.kalman_correction.abs().max() == pytest.approx(2)
    assert result.audit["governance_realised_days"] == 20
    assert result.audit["filter_parameters"]["governance_lookback_days"] == 60
